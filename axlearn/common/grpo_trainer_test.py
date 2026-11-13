import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import tempfile

import jax
import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import optimizers, test_utils
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.config import config_class, config_for_function
from axlearn.common.grpo_learner import AxlearnGrpoLearner
from axlearn.common.grpo_model import GrpoModel
from axlearn.common.grpo_trainer import GrpoSpmdTrainer
from axlearn.common.input_base import Input
from axlearn.common.learner import UpdateType


class DummyVocab:
    def instantiate(self):
        return self

    def decode(self, ids):
        if ids and ids[0] == 2:
            return "mock_decoded_text #### 99"
        return "mock_decoded_text #### 42"


class MockGRPOInput(Input):
    @config_class
    class Config(Input.Config):
        pass

    def dataset(self):
        # Yield multiple batches with 8 prompt tokens (<0) + 24 target tokens (>0)
        target_labels = jnp.concatenate(
            [-jnp.ones([8, 8], dtype=jnp.int32), jnp.ones([8, 24], dtype=jnp.int32)], axis=-1
        )
        for _ in range(8):
            batch = {
                "input_ids": jnp.ones([8, 32], dtype=jnp.int32),
                "target_labels": target_labels,
            }
            yield batch


class SimpleDecoder(BaseModel):
    @config_class
    class Config(BaseModel.Config):
        pad_token_id: int = 0

    def create_parameter_specs_recursively(self):
        return {
            "weight": ParameterSpec(
                shape=[8, 8],
                dtype=jnp.float32,
                mesh_axes=PartitionSpec("model"),
            )
        }

    def initialize_parameters_recursively(self, prng_key, *, prebuilt=None):
        return {"weight": jax.random.normal(prng_key, [8, 8], dtype=jnp.float32)}

    def sample_decode(self, input_batch, max_sequence_length, num_decodes):
        batch_size = input_batch["prefix"].shape[0]
        # Give sibling generations contrasting tokens (1 vs 2) so group advantages are non-zero!
        gen_tokens = jnp.arange(1, num_decodes + 1, dtype=jnp.int32)[None, :, None]
        seqs = jnp.broadcast_to(gen_tokens, [batch_size, num_decodes, max_sequence_length])

        class Outputs:
            sequences = seqs
            token_scores = -jnp.ones(
                [batch_size, num_decodes, max_sequence_length], dtype=jnp.float32
            )

        return Outputs()

    def project(self, x):
        return x @ self.state["weight"]


class MockModel(BaseModel):
    @config_class
    class Config(BaseModel.Config):
        pass

    def __init__(self, cfg: Config, *, parent):
        super().__init__(cfg, parent=parent)
        self._add_child("decoder", SimpleDecoder.default_config())

    def predict(self, input_batch):
        x = input_batch["input_ids"].astype(jnp.float32)  # Shape [batch, seq_len]
        seq_len = x.shape[1]
        # Enter child context of decoder to access its state
        from axlearn.common.module import child_context, new_output_collection

        output_collection = new_output_collection()
        with child_context(
            "decoder",
            module=self.decoder,
            state=self.state["decoder"],
            output_collection=output_collection,
        ):
            projected = self.decoder.project(x[:, :8])
        # Tile projected representation to match logits shape dynamically along the sequence axis
        logits = jnp.tile(projected[:, None, :], (1, seq_len, 1))[:, :, :10]
        return {"logits": logits}


class GrpoTrainerTest(test_utils.TestCase):
    """Tests GrpoSpmdTrainer disaggregated execution."""

    @parameterized.parameters(
        {
            "platform": "cpu",
            "mesh_shape": (1, 1, 1, 8, 1, 1),
            "reward_type": "gsm8k",
            "max_step": 3,
        },
        {
            "platform": "cpu",
            "mesh_shape": (1, 1, 1, 8, 1, 1),
            "reward_type": "dummy",
            "max_step": 1,
        },
        {
            "platform": "cpu",
            "mesh_shape": (1, 1, 1, 8, 1, 1),
            "reward_type": "exact_match",
            "max_step": 2,
        },
    )
    def test_disaggregated_trainer(self, platform, mesh_shape, reward_type, max_step):
        if not test_utils.is_supported_platform(platform):
            return

        # 1. Build config
        cfg = GrpoSpmdTrainer.default_config().set(
            name="grpo_trainer",
            dir=tempfile.mkdtemp(),
            mesh_axis_names=("pipeline", "data", "expert", "fsdp", "seq", "model"),
            mesh_shape=mesh_shape,
            reward_type=reward_type,
            model=GrpoModel.default_config().set(
                name="model",
                dtype=jnp.float32,
                actor=MockModel.default_config(),
                reference=MockModel.default_config(),
                sampler=MockModel.default_config(),
            ),
            vocab=config_for_function(DummyVocab),
            num_generations=2,
            max_step=max_step,
            learner=AxlearnGrpoLearner.default_config().set(
                name="learner",
                num_generations=2,
                beta=0.04,
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1, decouple_weight_decay=True, weight_decay=0.01
                ),
                update_rules=[
                    ("reference/.*", UpdateType.NO_UPDATE),
                    ("sampler/.*", UpdateType.NO_UPDATE),
                ],
            ),
            input=MockGRPOInput.default_config(),
        )

        # 2. Instantiate Trainer
        trainer: GrpoSpmdTrainer = cfg.instantiate(parent=None)

        # Verify dual meshes constructed on strictly disjoint physical device sets
        self.assertEqual(trainer.trainer_mesh.devices.size, 4)
        self.assertEqual(trainer.rollout_mesh.devices.size, 4)
        self.assertTrue(
            set(trainer.trainer_mesh.devices.flatten()).isdisjoint(
                set(trainer.rollout_mesh.devices.flatten())
            )
        )
        # Verify opposite shardings: FSDP on trainer_mesh vs TP (model=4, fsdp=1) on rollout_mesh
        self.assertEqual(trainer.trainer_mesh.shape["fsdp"], 4)
        self.assertEqual(trainer.trainer_mesh.shape["model"], 1)
        self.assertEqual(trainer.rollout_mesh.shape["fsdp"], 1)
        self.assertEqual(trainer.rollout_mesh.shape["model"], 4)

        # Verify Actor & Reference bind to trainer_mesh while Sampler is decoupled onto rollout_mesh
        self.assertEqual(
            trainer.trainer_state_partition_specs.model["actor"]["decoder"]["weight"].mesh,
            trainer.trainer_mesh,
        )
        self.assertEqual(
            trainer.trainer_state_partition_specs.model["reference"]["decoder"]["weight"].mesh,
            trainer.trainer_mesh,
        )
        self.assertNotIn("sampler", trainer.trainer_state_partition_specs.model)

        # 3. Initialize state on trainer_mesh
        prng_key = jax.random.PRNGKey(123)
        with trainer.mesh():
            trainer.init(prng_key)

        state = trainer.trainer_state

        # Capture initial parameters
        initial_actor_weight = np.asarray(state.model["actor"]["decoder"]["weight"]).copy()
        initial_ref_weight = np.asarray(state.model["reference"]["decoder"]["weight"]).copy()

        # 4. Execute training steps (including N+1 async rollout/update overlap)
        output = trainer.run(prng_key=prng_key)
        self.assertIsNotNone(output)
        self.assertEqual(trainer.step, max_step)
        final_state = trainer.trainer_state

        final_actor_weight = final_state.model["actor"]["decoder"]["weight"]
        final_ref_weight = final_state.model["reference"]["decoder"]["weight"]
        final_sampler_weight = trainer.sampler_params["decoder"]["weight"]

        # 5. Validate results
        # Actor should have updated weights on trainer_mesh
        self.assertFalse(np.array_equal(initial_actor_weight, np.asarray(final_actor_weight)))
        self.assertEqual(final_actor_weight.sharding.mesh, trainer.trainer_mesh)

        # Reference weights must be COMPLETELY unchanged on trainer_mesh
        self.assertTrue(np.array_equal(initial_ref_weight, np.asarray(final_ref_weight)))
        self.assertEqual(final_ref_weight.sharding.mesh, trainer.trainer_mesh)

        # Sampler weights must match final actor weights AND reside on rollout_mesh!
        self.assertTrue(
            np.allclose(np.asarray(final_sampler_weight), np.asarray(final_actor_weight))
        )
        self.assertEqual(final_sampler_weight.sharding.mesh, trainer.rollout_mesh)

        # Verify optimizer state is only allocated for actor (reference has none)
        optimizer_state = final_state.learner["optimizer"]
        import optax

        # Flatten the entire optimizer state tree to inspect dynamic leaves
        opt_items = test_utils.flatten_items(optimizer_state)
        for path, value in opt_items:
            if "reference" in path or "sampler" in path:
                self.assertTrue(value is None or isinstance(value, optax.MaskedNode))
            elif "actor" in path:
                self.assertFalse(value is None or isinstance(value, optax.MaskedNode))


if __name__ == "__main__":
    absltest.main()
