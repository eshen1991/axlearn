## Job Launch Command
```shell
export BASTION_TIER=disabled
export NAME=eshen-v6e-rl-grpo
export RUNNER_NAME=gke_tpu_single
export CONFIG=grpo-fuji-8B-v1
export INSTANCE_TYPE=tpu-v6e-16
export CLUSTER=ericshen-axlearn
export OUTPUT_DIR="gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/${NAME}/$(date +%s)"

(.venv) ➜  axlearn git:(rl-example) ✗ nohup axlearn gcp launch run \
  --cluster=$CLUSTER \
  --instance_type=$INSTANCE_TYPE \
  --name=$NAME \
  --num_replicas=1 \
  --runner_name=$RUNNER_NAME \
  --bundler_spec=allow_dirty=True \
  --bundler_type=artifactregistry \
  --bundler_spec=image=tpu \
  --bundler_spec=dockerfile=Dockerfile \
  --bundler_spec=target=tpu \
  -- \
  python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.grpo_native_example \
  --config=$CONFIG \
  --trainer_dir=$OUTPUT_DIR \
  --data_dir=gs://ericshen-axlearn/tensorflow-datasets \
  --jax_backend=tpu \
  --trace_at_steps=10 --trainer_log_every_n_steps=10  > axlearn_launcher.log 2>&1 &

```



## output directory gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo/1778611794/
## trainer_config
```shell
➜  ~ gcloud storage cat gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo/1778611794/trainer_config
batch_axis_names: 'data'
checkpointer.gc_loop_interval_seconds: 60
checkpointer.keep_last_n: 1
checkpointer.klass: 'axlearn.common.checkpointer.Checkpointer'
checkpointer.save_policy.fn: 'axlearn.common.checkpointer.every_n_steps_policy'
checkpointer.save_policy.min_step: 1
checkpointer.save_policy.n: 1
checkpointer.storage.klass: 'axlearn.common.checkpointer.TensorStoreStateStorage'
checkpointer.storage.timeout_secs: 3600
crash_on_hang_timeout_seconds: 7200
dir: 'gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo/1778611794'
init_state_builder.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOStateBuilder'
init_state_builder.storage_builder.concurrent_gb: 32
init_state_builder.storage_builder.dir: 'gs://ericshen-axlearn/checkpoints/llama-3-1-8B-instruct/step_00000000'
init_state_builder.storage_builder.klass: 'axlearn.common.state_builder.TensorStoreStateStorageBuilder'
init_state_builder.storage_builder.storage.klass: 'axlearn.common.checkpointer.TensorStoreStateStorage'
init_state_builder.storage_builder.storage.timeout_secs: 3600
init_state_builder.storage_builder.validation: 'CONTAINS_STATE_UP_TO_DTYPE'
input.batcher.fn: 'axlearn.common.input_tf_data.per_feed_batch'
input.batcher.pad_example_fn: 'axlearn.common.input_tf_data.default_pad_example_fn'
input.batcher.prefetch_buffer_size: -1
input.input_dispatcher.global_logical_batch_size: 128
input.input_dispatcher.klass: 'axlearn.common.input_dispatch.SpmdInputDispatcher'
input.input_dispatcher.partition_spec: PartitionSpec(('data', 'expert', 'fsdp'),)
input.input_partitioner.fn: 'axlearn.common.input_base.partition_by_path_rank'
input.input_partitioner.path_rank_to_partition[(None, 1)]: PartitionSpec(('data', 'expert', 'fsdp'),)
input.input_partitioner.path_rank_to_partition[(None, 2)]: PartitionSpec(('data', 'expert', 'fsdp'), 'seq')
input.is_training: True
input.klass: 'axlearn.common.input_tf_data.Input'
input.processor.fn: 'axlearn.common.input_tf_data.identity'
input.source.dataset_name: 'gsm8k'
input.source.fn: 'axlearn.experiments.text.gpt.grpo_native_example.gsm8k_tfds_input'
input.source.is_training: True
input.source.max_sequence_length: 512
input.source.split: 'train'
input.source.train_shuffle_buffer_size: 16384
input.source.vocab_cfg.filename: 'Llama-3-tokenizer.json'
input.source.vocab_cfg.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOV3Vocabulary'
klass: 'axlearn.common.grpo_trainer.GrpoSpmdTrainer'
learner.beta: 0.04
learner.ema.fn: 'axlearn.common.optimizers.param_ema'
learner.enable_per_variable_summaries: False
learner.epsilon: 0.2
learner.klass: 'axlearn.common.grpo_learner.AxlearnGrpoLearner'
learner.name: 'learner'
learner.num_generations: 2
learner.optimizer.b1: 0.9
learner.optimizer.b2: 0.95
learner.optimizer.eps: 1e-08
learner.optimizer.fn: 'axlearn.common.optimizers.adamw_optimizer'
learner.optimizer.learning_rate: 1e-05
learner.optimizer.weight_decay: 0
log_every_n_steps: 10
max_step: 1000
mesh_axis_names[0]: 'pipeline'
mesh_axis_names[1]: 'data'
mesh_axis_names[2]: 'expert'
mesh_axis_names[3]: 'fsdp'
mesh_axis_names[4]: 'seq'
mesh_axis_names[5]: 'model'
mesh_shape[0]: 1
mesh_shape[1]: 1
mesh_shape[2]: 1
mesh_shape[3]: 16
mesh_shape[4]: 1
mesh_shape[5]: 1
model.actor.batch_axis_names: None
model.actor.decoder.attention_mask: None
model.actor.decoder.decoding.klass: 'axlearn.common.decoder.DecodingLayer'
model.actor.decoder.dim: 4096
model.actor.decoder.dropout_rate: 0.0
model.actor.decoder.dtype: 'jax.numpy.bfloat16'
model.actor.decoder.emb.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.emb.klass: 'axlearn.common.embedding.TransformerTextEmbeddings'
model.actor.decoder.emb.token_emb.klass: 'axlearn.common.layers.Embedding'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].fan: 'fan_out'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.actor.decoder.emb.token_emb.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.actor.decoder.emb.token_emb.param_partition_spec[0]: 'model'
model.actor.decoder.emb.token_emb.param_partition_spec[1][0]: 'expert'
model.actor.decoder.emb.token_emb.param_partition_spec[1][1]: 'fsdp'
model.actor.decoder.emb.token_emb.param_partition_spec[1][2]: 'seq'
model.actor.decoder.eos_token_id: 128001
model.actor.decoder.klass: 'axlearn.common.decoder.Decoder'
model.actor.decoder.lm_head.klass: 'axlearn.common.decoder.LmHead'
model.actor.decoder.lm_head.param_partition_spec[0]: 'model'
model.actor.decoder.lm_head.param_partition_spec[1][0]: 'expert'
model.actor.decoder.lm_head.param_partition_spec[1][1]: 'fsdp'
model.actor.decoder.lm_head.param_partition_spec[1][2]: 'seq'
model.actor.decoder.logits_partition_spec[0][0]: 'data'
model.actor.decoder.logits_partition_spec[0][1]: 'expert'
model.actor.decoder.logits_partition_spec[0][2]: 'fsdp'
model.actor.decoder.logits_partition_spec[1]: 'seq'
model.actor.decoder.logits_partition_spec[2]: 'model'
model.actor.decoder.output_dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.output_norm.eps: 1e-05
model.actor.decoder.output_norm.forward_dtype: None
model.actor.decoder.output_norm.klass: 'axlearn.common.layers.RMSNorm'
model.actor.decoder.pad_token_id: 128004
model.actor.decoder.transformer.klass: 'axlearn.common.attention.RepeatedTransformerLayer'
model.actor.decoder.transformer.layer.feed_forward.activation[0]: 'nn.silu'
model.actor.decoder.transformer.layer.feed_forward.activation[1]: 'linear'
model.actor.decoder.transformer.layer.feed_forward.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.transformer.layer.feed_forward.hidden_dim.fn: 'axlearn.experiments.text.gpt.common.scale_fn'
model.actor.decoder.transformer.layer.feed_forward.hidden_dim.round_up_to_multiples_of: 256
model.actor.decoder.transformer.layer.feed_forward.hidden_dim.scale: 3.5
model.actor.decoder.transformer.layer.feed_forward.klass: 'axlearn.common.attention.TransformerFeedForwardLayer'
model.actor.decoder.transformer.layer.feed_forward.linear1.bias: False
model.actor.decoder.transformer.layer.feed_forward.linear1.klass: 'axlearn.common.layers.Linear'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][0]: 'data'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][1]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][2]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[1]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[2]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][0]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][1]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][2]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[1]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear2.bias: False
model.actor.decoder.transformer.layer.feed_forward.linear2.klass: 'axlearn.common.layers.Linear'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][0]: 'data'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][1]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][2]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[1]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[2]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[0]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][0]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][1]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][2]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.norm.eps: 1e-05
model.actor.decoder.transformer.layer.feed_forward.norm.forward_dtype: None
model.actor.decoder.transformer.layer.feed_forward.norm.klass: 'axlearn.common.layers.RMSNorm'
model.actor.decoder.transformer.layer.feed_forward.residual_weight: 1.0
model.actor.decoder.transformer.layer.feed_forward.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.actor.decoder.transformer.layer.feed_forward.stochastic_depth.mode: 'row'
model.actor.decoder.transformer.layer.feed_forward.structure: 'prenorm'
model.actor.decoder.transformer.layer.klass: 'axlearn.common.attention.TransformerLayer'
model.actor.decoder.transformer.layer.remat_spec['prevent_cse']: False
model.actor.decoder.transformer.layer.remat_spec['policy'].fn: 'axlearn.common.attention._save_and_offload_only_these_names_regex'
model.actor.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_offloaded: None
model.actor.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: '.*([qkvo]_proj|context)'
model.actor.decoder.transformer.layer.remat_spec['policy'].offload_dst: 'pinned_host'
model.actor.decoder.transformer.layer.remat_spec['policy'].offload_src: 'device'
model.actor.decoder.transformer.layer.self_attention.attention.causal: True
model.actor.decoder.transformer.layer.self_attention.attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.klass: 'axlearn.common.attention.FusedGroupedQKVLinear'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.bias: False
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.klass: 'axlearn.common.attention.MultiheadInputLinear'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][0]: 'expert'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][1]: 'fsdp'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][2]: 'seq'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[1]: 'model'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[2]: None
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.num_kv_heads: 8
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.klass: 'axlearn.common.attention.RoFormerQKVLinear'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.klass: 'axlearn.common.attention.RoFormerSinusoidalPositionalEmbedding'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.theta: 500000.0
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.rotary_value: False
model.actor.decoder.transformer.layer.self_attention.attention.key_scale.klass: 'axlearn.common.attention.ScaleKey'
model.actor.decoder.transformer.layer.self_attention.attention.klass: 'axlearn.common.attention.GroupedQueryAttention'
model.actor.decoder.transformer.layer.self_attention.attention.kv_cache.cache_dtype: 'jax.numpy.bfloat16'
model.actor.decoder.transformer.layer.self_attention.attention.kv_cache.klass: 'axlearn.common.kv_cache.kv_cache.KVCache'
model.actor.decoder.transformer.layer.self_attention.attention.num_heads: 32
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.bias: False
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.klass: 'axlearn.common.attention.MultiheadOutputLinear'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][0]: 'expert'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][1]: 'fsdp'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][2]: 'seq'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[1]: 'model'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[2]: None
model.actor.decoder.transformer.layer.self_attention.attention.query_scale.klass: 'axlearn.common.attention.ScaleQuery'
model.actor.decoder.transformer.layer.self_attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.transformer.layer.self_attention.klass: 'axlearn.common.attention.TransformerAttentionLayer'
model.actor.decoder.transformer.layer.self_attention.norm.eps: 1e-05
model.actor.decoder.transformer.layer.self_attention.norm.forward_dtype: None
model.actor.decoder.transformer.layer.self_attention.norm.klass: 'axlearn.common.layers.RMSNorm'
model.actor.decoder.transformer.layer.self_attention.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.actor.decoder.transformer.layer.self_attention.stochastic_depth.mode: 'row'
model.actor.decoder.transformer.layer.self_attention.structure: 'prenorm'
model.actor.decoder.transformer.num_layers: 32
model.actor.decoder.transformer.repeat.drop_output.fn: 'axlearn.common.repeat._drop_by_regex'
model.actor.decoder.transformer.repeat.drop_output.rules[0]: 'module_outputs.*'
model.actor.decoder.transformer.repeat.klass: 'axlearn.common.attention._TransformerRepeat'
model.actor.decoder.vocab_size: 128256
model.actor.dtype: 'jax.numpy.bfloat16'
model.actor.klass: 'axlearn.common.causal_lm.Model'
model.actor.metrics.klass: 'axlearn.common.causal_lm.CompositeLossMetrics'
model.actor.metrics.metrics['lm'].klass: 'axlearn.common.causal_lm.CrossEntropyLossMetrics'
model.actor.metrics.metrics['aux'].klass: 'axlearn.common.causal_lm.AuxLossMetrics'
model.actor.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.actor.param_init.init_by_param_name['.*weight$'].fan: 'fan_in'
model.actor.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.actor.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.actor.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.dtype: 'jax.numpy.bfloat16'
model.klass: 'axlearn.common.grpo_model.GrpoModel'
model.name: 'model'
model.reference.batch_axis_names: None
model.reference.decoder.attention_mask: None
model.reference.decoder.decoding.klass: 'axlearn.common.decoder.DecodingLayer'
model.reference.decoder.dim: 4096
model.reference.decoder.dropout_rate: 0.0
model.reference.decoder.dtype: 'jax.numpy.bfloat16'
model.reference.decoder.emb.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.emb.klass: 'axlearn.common.embedding.TransformerTextEmbeddings'
model.reference.decoder.emb.token_emb.klass: 'axlearn.common.layers.Embedding'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].fan: 'fan_out'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.reference.decoder.emb.token_emb.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.reference.decoder.emb.token_emb.param_partition_spec[0]: 'model'
model.reference.decoder.emb.token_emb.param_partition_spec[1][0]: 'expert'
model.reference.decoder.emb.token_emb.param_partition_spec[1][1]: 'fsdp'
model.reference.decoder.emb.token_emb.param_partition_spec[1][2]: 'seq'
model.reference.decoder.eos_token_id: 128001
model.reference.decoder.klass: 'axlearn.common.decoder.Decoder'
model.reference.decoder.lm_head.klass: 'axlearn.common.decoder.LmHead'
model.reference.decoder.lm_head.param_partition_spec[0]: 'model'
model.reference.decoder.lm_head.param_partition_spec[1][0]: 'expert'
model.reference.decoder.lm_head.param_partition_spec[1][1]: 'fsdp'
model.reference.decoder.lm_head.param_partition_spec[1][2]: 'seq'
model.reference.decoder.logits_partition_spec[0][0]: 'data'
model.reference.decoder.logits_partition_spec[0][1]: 'expert'
model.reference.decoder.logits_partition_spec[0][2]: 'fsdp'
model.reference.decoder.logits_partition_spec[1]: 'seq'
model.reference.decoder.logits_partition_spec[2]: 'model'
model.reference.decoder.output_dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.output_norm.eps: 1e-05
model.reference.decoder.output_norm.forward_dtype: None
model.reference.decoder.output_norm.klass: 'axlearn.common.layers.RMSNorm'
model.reference.decoder.pad_token_id: 128004
model.reference.decoder.transformer.klass: 'axlearn.common.attention.RepeatedTransformerLayer'
model.reference.decoder.transformer.layer.feed_forward.activation[0]: 'nn.silu'
model.reference.decoder.transformer.layer.feed_forward.activation[1]: 'linear'
model.reference.decoder.transformer.layer.feed_forward.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.transformer.layer.feed_forward.hidden_dim.fn: 'axlearn.experiments.text.gpt.common.scale_fn'
model.reference.decoder.transformer.layer.feed_forward.hidden_dim.round_up_to_multiples_of: 256
model.reference.decoder.transformer.layer.feed_forward.hidden_dim.scale: 3.5
model.reference.decoder.transformer.layer.feed_forward.klass: 'axlearn.common.attention.TransformerFeedForwardLayer'
model.reference.decoder.transformer.layer.feed_forward.linear1.bias: False
model.reference.decoder.transformer.layer.feed_forward.linear1.klass: 'axlearn.common.layers.Linear'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][0]: 'data'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][1]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][2]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[1]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[2]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][0]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][1]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][2]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[1]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear2.bias: False
model.reference.decoder.transformer.layer.feed_forward.linear2.klass: 'axlearn.common.layers.Linear'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][0]: 'data'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][1]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][2]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[1]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[2]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[0]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][0]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][1]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][2]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.norm.eps: 1e-05
model.reference.decoder.transformer.layer.feed_forward.norm.forward_dtype: None
model.reference.decoder.transformer.layer.feed_forward.norm.klass: 'axlearn.common.layers.RMSNorm'
model.reference.decoder.transformer.layer.feed_forward.residual_weight: 1.0
model.reference.decoder.transformer.layer.feed_forward.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.reference.decoder.transformer.layer.feed_forward.stochastic_depth.mode: 'row'
model.reference.decoder.transformer.layer.feed_forward.structure: 'prenorm'
model.reference.decoder.transformer.layer.klass: 'axlearn.common.attention.TransformerLayer'
model.reference.decoder.transformer.layer.remat_spec['prevent_cse']: False
model.reference.decoder.transformer.layer.remat_spec['policy'].fn: 'axlearn.common.attention._save_and_offload_only_these_names_regex'
model.reference.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_offloaded: None
model.reference.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: '.*([qkvo]_proj|context)'
model.reference.decoder.transformer.layer.remat_spec['policy'].offload_dst: 'pinned_host'
model.reference.decoder.transformer.layer.remat_spec['policy'].offload_src: 'device'
model.reference.decoder.transformer.layer.self_attention.attention.causal: True
model.reference.decoder.transformer.layer.self_attention.attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.klass: 'axlearn.common.attention.FusedGroupedQKVLinear'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.bias: False
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.klass: 'axlearn.common.attention.MultiheadInputLinear'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][0]: 'expert'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][1]: 'fsdp'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][2]: 'seq'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[1]: 'model'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[2]: None
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.num_kv_heads: 8
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.klass: 'axlearn.common.attention.RoFormerQKVLinear'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.klass: 'axlearn.common.attention.RoFormerSinusoidalPositionalEmbedding'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.theta: 500000.0
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.rotary_value: False
model.reference.decoder.transformer.layer.self_attention.attention.key_scale.klass: 'axlearn.common.attention.ScaleKey'
model.reference.decoder.transformer.layer.self_attention.attention.klass: 'axlearn.common.attention.GroupedQueryAttention'
model.reference.decoder.transformer.layer.self_attention.attention.kv_cache.cache_dtype: 'jax.numpy.bfloat16'
model.reference.decoder.transformer.layer.self_attention.attention.kv_cache.klass: 'axlearn.common.kv_cache.kv_cache.KVCache'
model.reference.decoder.transformer.layer.self_attention.attention.num_heads: 32
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.bias: False
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.klass: 'axlearn.common.attention.MultiheadOutputLinear'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][0]: 'expert'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][1]: 'fsdp'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][2]: 'seq'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[1]: 'model'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[2]: None
model.reference.decoder.transformer.layer.self_attention.attention.query_scale.klass: 'axlearn.common.attention.ScaleQuery'
model.reference.decoder.transformer.layer.self_attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.transformer.layer.self_attention.klass: 'axlearn.common.attention.TransformerAttentionLayer'
model.reference.decoder.transformer.layer.self_attention.norm.eps: 1e-05
model.reference.decoder.transformer.layer.self_attention.norm.forward_dtype: None
model.reference.decoder.transformer.layer.self_attention.norm.klass: 'axlearn.common.layers.RMSNorm'
model.reference.decoder.transformer.layer.self_attention.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.reference.decoder.transformer.layer.self_attention.stochastic_depth.mode: 'row'
model.reference.decoder.transformer.layer.self_attention.structure: 'prenorm'
model.reference.decoder.transformer.num_layers: 32
model.reference.decoder.transformer.repeat.drop_output.fn: 'axlearn.common.repeat._drop_by_regex'
model.reference.decoder.transformer.repeat.drop_output.rules[0]: 'module_outputs.*'
model.reference.decoder.transformer.repeat.klass: 'axlearn.common.attention._TransformerRepeat'
model.reference.decoder.vocab_size: 128256
model.reference.dtype: 'jax.numpy.bfloat16'
model.reference.klass: 'axlearn.common.causal_lm.Model'
model.reference.metrics.klass: 'axlearn.common.causal_lm.CompositeLossMetrics'
model.reference.metrics.metrics['lm'].klass: 'axlearn.common.causal_lm.CrossEntropyLossMetrics'
model.reference.metrics.metrics['aux'].klass: 'axlearn.common.causal_lm.AuxLossMetrics'
model.reference.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.reference.param_init.init_by_param_name['.*weight$'].fan: 'fan_in'
model.reference.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.reference.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.reference.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
name: 'grpo_trainer'
num_generations: 2
prune_empty_state_updates: True
recorder.fn: '__main__.<lambda>'
save_input_iterator: False
start_trace_process_indices[0]: 0
start_trace_steps[0]: 10
summary_writer.klass: 'axlearn.common.summary_writer.SummaryWriter'
summary_writer.max_queue: 1000
summary_writer.write_every_n_steps: 1
vocab.filename: 'Llama-3-tokenizer.json'
vocab.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOV3Vocabulary'
```
## model_analysis
```shell
(.venv) ➜  axlearn git:(rl-example-pathways) ✗ gcloud storage cat gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo/1778611794/model_analysis.txt | tr '\t' '\n'
##################### Model analysis #####################
## Parameters:
 525336576 [128256, 4096]       actor/decoder/emb/token_emb/weight
 525336576 (128256, 4096)       actor/decoder/lm_head/weight
      4096 [4096]               actor/decoder/output_norm/scale
1879048192 (32, 4096, 14336)    actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight
1879048192 (32, 4096, 14336)    actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight
1879048192 (32, 14336, 4096)    actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight
    131072 (32, 4096)           actor/decoder/transformer/repeat/layer/feed_forward/norm/scale
 805306368 (32, 4096, 48, 128)  actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight
 536870912 (32, 4096, 32, 128)  actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight
    131072 (32, 4096)           actor/decoder/transformer/repeat/layer/self_attention/norm/scale
 525336576 [128256, 4096]       reference/decoder/emb/token_emb/weight
 525336576 (128256, 4096)       reference/decoder/lm_head/weight
      4096 [4096]               reference/decoder/output_norm/scale
1879048192 (32, 4096, 14336)    reference/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight
1879048192 (32, 4096, 14336)    reference/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight
1879048192 (32, 14336, 4096)    reference/decoder/transformer/repeat/layer/feed_forward/linear2/weight
    131072 (32, 4096)           reference/decoder/transformer/repeat/layer/feed_forward/norm/scale
 805306368 (32, 4096, 48, 128)  reference/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight
 536870912 (32, 4096, 32, 128)  reference/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight
    131072 (32, 4096)           reference/decoder/transformer/repeat/layer/self_attention/norm/scale
Total number of model params: 16,060,522,496
## Trainer States:
State: prng_key=uint32((4,)) mesh_axes=ParameterSpec(shape=[4], dtype=<class 'jax.numpy.uint32'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/actor/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/actor/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/actor/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=ParameterSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=ParameterSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(1,), out_axis=(2, 3), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(2, 3), out_axis=(1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/reference/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/reference/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/reference/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=ParameterSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=ParameterSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(1,), out_axis=(2, 3), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(2, 3), out_axis=(1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: learner/optimizer/0/count=int32(()) mesh_axes=TensorSpec(shape=(), dtype=dtype('int32'), mesh_axes=PartitionSpec(), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=TensorSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=TensorSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=TensorSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=TensorSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/mu/reference/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=TensorSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=TensorSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=TensorSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=TensorSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/nu/reference/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/2/count=int32(()) mesh_axes=TensorSpec(shape=[], dtype=<class 'jax.numpy.int32'>, mesh_axes=PartitionSpec(), memory_kind=None)
Training state size: 89.75 GiB
Training state size (partitioned): 5.61 GiB
Max training state size (partitioned): 5.61 GiB
##########################################################
```
## Training Metrics

### Definition

#### 1. **`loss`** (Policy Gradient Objective)
*   **Definition**: The overall composite objective minimized by the optimizer. It combines the negative GRPO surrogate policy loss (which pushes the Actor to produce higher-advantage tokens) and the KL divergence penalty.
*   **Plain-Text Formula**:
    `Loss = -(Average of [ Minimum( Ratio * Advantage, Clipped_Ratio * Advantage ) ]) + beta * KL_Divergence`
*   **Intuition**: This is the "steering wheel" of your training. A smaller/more negative loss indicates the Actor model is successfully optimizing its parameters to maximize mathematical rewards while remaining safely constrained.

---

#### 2. **`kl_divergence`** (Kullback-Leibler Divergence)
*   **Definition**: A statistical measure of how much the active Actor model's token probability distribution has drifted/diverged from the frozen, pre-trained Reference model's token probability distribution.
*   **Plain-Text Formula**:
    `KL_Divergence = Average of [ (Reference_Probability / Actor_Probability) - log(Reference_Probability / Actor_Probability) - 1 ]`
*   **Intuition**: This measures the Actor's "deviation penalty." We want this to remain stable and small (e.g. `0.002 - 0.005`). If it surges too high, the model is "forgetting" its pre-trained knowledge or collapsing into gibberish to exploit the rewards.

---

#### 3. **`mean_advantage`**
*   **Definition**: The average relative advantage score across all generations in the batch.
*   **Plain-Text Formula**:
    `Mean_Advantage = (Sum of all Advantages in Batch) / (Total Number of Rollouts in Batch) = 0.0`
*   **Intuition**: Because GRPO computes relative advantages by Z-score normalizing the rewards *inside* each sibling group, the above-average advantages perfectly cancel out the below-average advantages. Thus, this metric is **mathematically, strictly always exactly `0.0`** by design.

---

#### 4. **`mean_reward`** (Absolute Scoreboard Accuracy)
*   **Definition**: The percentage of generated rollouts in the current training batch that successfully solved the math problem correctly and matched the ground-truth final answer (received a reward of `1.0`).
*   **Plain-Text Formula**:
    `Mean_Reward = (Number of Correct Rollouts in Batch) / (Total Number of Rollouts in Batch)`
*   **Intuition**: This is the **absolute scoreboard** showing how smart your model is in the real world. A value of `0.714` means exactly **71.4% of all math questions solved in this batch were completely correct.**

---

#### 5. **`std_advantage`** (The Learning Signal Contrast)
*   **Definition**: The standard deviation (the spread/variance) of the relative advantages. It measures the **strength and presence of contrast** inside your training groups.
*   **Plain-Text Formula**:
    `Std_Advantage = Standard_Deviation(Advantages_in_Batch)`
*   **Intuition**: Think of this as the **learning signal strength**.
    *   If it is positive ($>0.0$): It confirms that in some groups, some siblings were correct and others were incorrect. This **contrast** provides a strong active gradient signal to the Actor model.
    *   If it were $0.0$: All siblings got the exact same reward (no contrast), and the model would learn nothing.

### Step Time Calculation
#### Overall Average Calculation

The training run started at **2026-05-12 12:07:25** ([appl rl grpo training ...](https://paste.googleplex.com/6286208761200640?content_ref=2026+05+12+12+07+25+816+grpo_trainer+process+0+step+1)) and reached Step 1000 at **2026-05-13 07:17:45** ([appl rl grpo training ...](https://paste.googleplex.com/6286208761200640?content_ref=2026+05+13+07+17+45+600+grpo_trainer+process+0+step+1000)).

*   **Total Elapsed Time:** approximately $19.17$ hours ($69,019.78$ seconds) ([appl rl grpo training ...](https://paste.googleplex.com/6286208761200640?content_ref=total+duration+1000+steps+19+17hr)).
*   **Total Steps Completed:** $999$ steps (from Step 1 to Step 1000).
*   **Average Step Time (Wall-Clock):** **1.15 minutes per step** (or $69.09$ seconds).

#### Active Training Time
For this specific run, the training appears highly consistent with no significant downtime between the logged steps ([appl rl grpo training ...](https://paste.googleplex.com/6286208761200640?content_ref=avg+duration+per+step+69+02s)).

| Metric | Duration (Minutes) | Duration (Seconds) |
| :--- | :---: | :---: |
| **Wall-Clock Average** | **1.15** | **69.09** |
| Active Training Average | 1.15 | 69.09 |

### Metrics Log

| Timestamp               | Message                                                                                                                                                                                                               |
|:------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-05-12 12:07:25.816 | grpo_trainer process 0 step 1] loss=-4.131498e-06 aux={'kl_divergence': 0.000659942626953125, 'loss': -4.131497917114757e-06, 'mean_advantage': 0.0, 'mean_reward': 0.71484375, 'std_advantage': 0.39025717973709106} |
| 2026-05-12 12:08:34.866 | grpo_trainer process 0 step 2] loss=0.00016860577 aux={'kl_divergence': 0.00421142578125, 'loss': 0.00016860576579347253, 'mean_advantage': 0.0, 'mean_reward': 0.765625, 'std_advantage': 0.3186436593532562}        |
| 2026-05-12 12:09:41.341 | grpo_trainer process 0 step 3] loss=0.0004939452 aux={'kl_divergence': 0.01287841796875, 'loss': 0.0004939452046528459, 'mean_advantage': 0.0, 'mean_reward': 0.76171875, 'std_advantage': 0.31245583295822144}       |
| 2026-05-12 12:10:47.367 | grpo_trainer process 0 step 4] loss=0.0003354485 aux={'kl_divergence': 0.0081787109375, 'loss': 0.0003354485088493675, 'mean_advantage': 0.0, 'mean_reward': 0.76953125, 'std_advantage': 0.35898441076278687}        |
| 2026-05-12 12:11:49.178 | grpo_trainer process 0 step 5] loss=0.00035129616 aux={'kl_divergence': 0.00860595703125, 'loss': 0.0003512961557134986, 'mean_advantage': 0.0, 'mean_reward': 0.76953125, 'std_advantage': 0.33652520179748535}      |
| 2026-05-12 12:17:12.748 | grpo_trainer process 0 step 10] loss=0.0007302054 aux={'kl_divergence': 0.0184326171875, 'loss': 0.0007302053854800761, 'mean_advantage': 0.0, 'mean_reward': 0.796875, 'std_advantage': 0.3852214515209198}          |
| 2026-05-12 12:28:56.627 | grpo_trainer process 0 step 20] loss=0.00053647865 aux={'kl_divergence': 0.013671875, 'loss': 0.0005364786484278738, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.35350340604782104}           |
| 2026-05-12 12:39:49.833 | grpo_trainer process 0 step 30] loss=0.0006506553 aux={'kl_divergence': 0.01611328125, 'loss': 0.0006506553036160767, 'mean_advantage': 0.0, 'mean_reward': 0.80859375, 'std_advantage': 0.34793609380722046}         |
| 2026-05-12 12:50:32.582 | grpo_trainer process 0 step 40] loss=0.0004379745 aux={'kl_divergence': 0.01080322265625, 'loss': 0.00043797449325211346, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.3061429262161255}       |
| 2026-05-12 13:01:19.370 | grpo_trainer process 0 step 50] loss=0.00048297076 aux={'kl_divergence': 0.0120849609375, 'loss': 0.00048297076136805117, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.2576576769351959}      |
| 2026-05-12 13:12:41.656 | grpo_trainer process 0 step 60] loss=0.00065352337 aux={'kl_divergence': 0.016357421875, 'loss': 0.000653523369692266, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.3061429262161255}          |
| 2026-05-12 13:23:49.504 | grpo_trainer process 0 step 70] loss=0.0013675039 aux={'kl_divergence': 0.0341796875, 'loss': 0.0013675038935616612, 'mean_advantage': 0.0, 'mean_reward': 0.875, 'std_advantage': 0.3061429262161255}                |
| 2026-05-12 13:34:51.334 | grpo_trainer process 0 step 80] loss=0.00083317445 aux={'kl_divergence': 0.0201416015625, 'loss': 0.0008331744465976954, 'mean_advantage': 0.0, 'mean_reward': 0.8125, 'std_advantage': 0.3186436593532562}           |
| 2026-05-12 13:45:58.692 | grpo_trainer process 0 step 90] loss=0.00086255895 aux={'kl_divergence': 0.021728515625, 'loss': 0.0008625589543953538, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.28637048602104187}       |
| 2026-05-12 13:57:12.921 | grpo_trainer process 0 step 100] loss=0.0007402969 aux={'kl_divergence': 0.0185546875, 'loss': 0.0007402969058603048, 'mean_advantage': 0.0, 'mean_reward': 0.796875, 'std_advantage': 0.3422781825065613}            |
| 2026-05-12 14:08:34.483 | grpo_trainer process 0 step 110] loss=0.001407758 aux={'kl_divergence': 0.034912109375, 'loss': 0.0014077579835429788, 'mean_advantage': 0.0, 'mean_reward': 0.7578125, 'std_advantage': 0.3186436593532562}          |
| 2026-05-12 14:19:47.366 | grpo_trainer process 0 step 120] loss=0.0011490695 aux={'kl_divergence': 0.02880859375, 'loss': 0.0011490695178508759, 'mean_advantage': 0.0, 'mean_reward': 0.75, 'std_advantage': 0.33067217469215393}              |
| 2026-05-12 14:30:56.347 | grpo_trainer process 0 step 130] loss=0.0011019479 aux={'kl_divergence': 0.02685546875, 'loss': 0.0011019478552043438, 'mean_advantage': 0.0, 'mean_reward': 0.72265625, 'std_advantage': 0.36970269680023193}        |
| 2026-05-12 14:42:18.278 | grpo_trainer process 0 step 140] loss=0.0011138218 aux={'kl_divergence': 0.028076171875, 'loss': 0.0011138217523694038, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.34793609380722046}       |
| 2026-05-12 14:53:52.202 | grpo_trainer process 0 step 150] loss=0.00085553067 aux={'kl_divergence': 0.021240234375, 'loss': 0.0008555306703783572, 'mean_advantage': 0.0, 'mean_reward': 0.73046875, 'std_advantage': 0.36970269680023193}      |
| 2026-05-12 15:05:10.994 | grpo_trainer process 0 step 160] loss=0.0009592255 aux={'kl_divergence': 0.0238037109375, 'loss': 0.0009592255228199065, 'mean_advantage': 0.0, 'mean_reward': 0.7734375, 'std_advantage': 0.3186436593532562}        |
| 2026-05-12 15:16:24.997 | grpo_trainer process 0 step 170] loss=0.0017815507 aux={'kl_divergence': 0.04443359375, 'loss': 0.0017815507017076015, 'mean_advantage': 0.0, 'mean_reward': 0.77734375, 'std_advantage': 0.28637048602104187}        |
| 2026-05-12 15:27:43.835 | grpo_trainer process 0 step 180] loss=0.0012910411 aux={'kl_divergence': 0.032958984375, 'loss': 0.0012910411460325122, 'mean_advantage': 0.0, 'mean_reward': 0.77734375, 'std_advantage': 0.2996971011161804}        |
| 2026-05-12 15:39:00.451 | grpo_trainer process 0 step 190] loss=0.0016261841 aux={'kl_divergence': 0.040771484375, 'loss': 0.0016261840937659144, 'mean_advantage': 0.0, 'mean_reward': 0.75390625, 'std_advantage': 0.2996971011161804}        |
| 2026-05-12 15:50:17.888 | grpo_trainer process 0 step 200] loss=0.003625129 aux={'kl_divergence': 0.09033203125, 'loss': 0.003625128883868456, 'mean_advantage': 0.0, 'mean_reward': 0.8125, 'std_advantage': 0.3749469816684723}               |
| 2026-05-12 16:01:38.290 | grpo_trainer process 0 step 210] loss=0.0042019007 aux={'kl_divergence': 0.10498046875, 'loss': 0.00420190067961812, 'mean_advantage': 0.0, 'mean_reward': 0.90625, 'std_advantage': 0.23382051289081573}             |
| 2026-05-12 16:13:10.368 | grpo_trainer process 0 step 220] loss=0.0026879753 aux={'kl_divergence': 0.0673828125, 'loss': 0.00268797529861331, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.2723926901817322}            |
| 2026-05-12 16:24:34.350 | grpo_trainer process 0 step 230] loss=0.003381556 aux={'kl_divergence': 0.0849609375, 'loss': 0.003381555899977684, 'mean_advantage': 0.0, 'mean_reward': 0.8046875, 'std_advantage': 0.3061429262161255}             |
| 2026-05-12 16:35:57.325 | grpo_trainer process 0 step 240] loss=0.0038848901 aux={'kl_divergence': 0.0966796875, 'loss': 0.003884890116751194, 'mean_advantage': 0.0, 'mean_reward': 0.8046875, 'std_advantage': 0.2931095361709595}            |
| 2026-05-12 16:47:25.774 | grpo_trainer process 0 step 250] loss=0.0025772 aux={'kl_divergence': 0.064453125, 'loss': 0.0025772000662982464, 'mean_advantage': 0.0, 'mean_reward': 0.796875, 'std_advantage': 0.27946898341178894}               |
| 2026-05-12 16:58:58.203 | grpo_trainer process 0 step 260] loss=0.0021326516 aux={'kl_divergence': 0.052978515625, 'loss': 0.0021326516289263964, 'mean_advantage': 0.0, 'mean_reward': 0.76953125, 'std_advantage': 0.28637048602104187}       |
| 2026-05-12 17:10:36.812 | grpo_trainer process 0 step 270] loss=0.0031445487 aux={'kl_divergence': 0.07958984375, 'loss': 0.0031445487402379513, 'mean_advantage': 0.0, 'mean_reward': 0.80859375, 'std_advantage': 0.34793609380722046}        |
| 2026-05-12 17:22:11.958 | grpo_trainer process 0 step 280] loss=0.0024655124 aux={'kl_divergence': 0.06201171875, 'loss': 0.0024655123706907034, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.31245583295822144}        |
| 2026-05-12 17:33:51.506 | grpo_trainer process 0 step 290] loss=0.0039598057 aux={'kl_divergence': 0.09912109375, 'loss': 0.003959805704653263, 'mean_advantage': 0.0, 'mean_reward': 0.76171875, 'std_advantage': 0.2723926901817322}          |
| 2026-05-12 17:45:28.683 | grpo_trainer process 0 step 300] loss=0.0032236967 aux={'kl_divergence': 0.08056640625, 'loss': 0.0032236967235803604, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.3422781825065613}          |
| 2026-05-12 17:57:04.975 | grpo_trainer process 0 step 310] loss=0.003588815 aux={'kl_divergence': 0.08984375, 'loss': 0.003588814986869693, 'mean_advantage': 0.0, 'mean_reward': 0.796875, 'std_advantage': 0.2931095361709595}                |
| 2026-05-12 18:08:47.172 | grpo_trainer process 0 step 320] loss=0.0019301112 aux={'kl_divergence': 0.0478515625, 'loss': 0.0019301112042739987, 'mean_advantage': 0.0, 'mean_reward': 0.796875, 'std_advantage': 0.3061429262161255}            |
| 2026-05-12 18:20:36.601 | grpo_trainer process 0 step 330] loss=0.002354327 aux={'kl_divergence': 0.05859375, 'loss': 0.002354326890781522, 'mean_advantage': 0.0, 'mean_reward': 0.7578125, 'std_advantage': 0.3061429262161255}               |
| 2026-05-12 18:32:22.922 | grpo_trainer process 0 step 340] loss=0.0032732864 aux={'kl_divergence': 0.08154296875, 'loss': 0.0032732863910496235, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.2931095361709595}           |
| 2026-05-12 18:44:09.647 | grpo_trainer process 0 step 350] loss=0.0016025743 aux={'kl_divergence': 0.0400390625, 'loss': 0.0016025742515921593, 'mean_advantage': 0.0, 'mean_reward': 0.8671875, 'std_advantage': 0.27946898341178894}          |
| 2026-05-12 18:55:57.456 | grpo_trainer process 0 step 360] loss=0.0018511494 aux={'kl_divergence': 0.046142578125, 'loss': 0.0018511493690311909, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.35350340604782104}        |
| 2026-05-12 19:07:49.367 | grpo_trainer process 0 step 370] loss=0.0006790894 aux={'kl_divergence': 0.0169677734375, 'loss': 0.0006790893967263401, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.35350340604782104}       |
| 2026-05-12 19:19:37.413 | grpo_trainer process 0 step 380] loss=0.00095713994 aux={'kl_divergence': 0.0250244140625, 'loss': 0.0009571399423293769, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.27946898341178894}      |
| 2026-05-12 19:31:39.568 | grpo_trainer process 0 step 390] loss=0.0010740956 aux={'kl_divergence': 0.02685546875, 'loss': 0.0010740956058725715, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.31245583295822144}        |
| 2026-05-12 19:43:36.506 | grpo_trainer process 0 step 400] loss=0.0015553297 aux={'kl_divergence': 0.03955078125, 'loss': 0.001555329654365778, 'mean_advantage': 0.0, 'mean_reward': 0.78125, 'std_advantage': 0.3186436593532562}             |
| 2026-05-12 19:55:28.564 | grpo_trainer process 0 step 410] loss=0.0018228914 aux={'kl_divergence': 0.044921875, 'loss': 0.0018228914123028517, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.2931095361709595}            |
| 2026-05-12 20:07:26.739 | grpo_trainer process 0 step 420] loss=0.0012881581 aux={'kl_divergence': 0.0322265625, 'loss': 0.0012881581205874681, 'mean_advantage': 0.0, 'mean_reward': 0.85546875, 'std_advantage': 0.33652520179748535}         |
| 2026-05-12 20:19:28.321 | grpo_trainer process 0 step 430] loss=0.0014549616 aux={'kl_divergence': 0.036376953125, 'loss': 0.001454961602576077, 'mean_advantage': 0.0, 'mean_reward': 0.8125, 'std_advantage': 0.27946901321411133}            |
| 2026-05-12 20:31:29.559 | grpo_trainer process 0 step 440] loss=0.0023019987 aux={'kl_divergence': 0.057373046875, 'loss': 0.002301998669281602, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.23382051289081573}          |
| 2026-05-12 20:43:29.591 | grpo_trainer process 0 step 450] loss=0.0008522521 aux={'kl_divergence': 0.0216064453125, 'loss': 0.0008522521238774061, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.2931095361709595}        |
| 2026-05-12 20:55:32.070 | grpo_trainer process 0 step 460] loss=0.003968836 aux={'kl_divergence': 0.09912109375, 'loss': 0.003968835808336735, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.33067217469215393}           |
| 2026-05-12 21:07:31.684 | grpo_trainer process 0 step 470] loss=0.014861636 aux={'kl_divergence': 0.37109375, 'loss': 0.014861635863780975, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.22531509399414062}             |
| 2026-05-12 21:19:31.401 | grpo_trainer process 0 step 480] loss=0.0046697543 aux={'kl_divergence': 0.1162109375, 'loss': 0.004669754300266504, 'mean_advantage': 0.0, 'mean_reward': 0.875, 'std_advantage': 0.27946898341178894}               |
| 2026-05-12 21:31:25.882 | grpo_trainer process 0 step 490] loss=0.0013640915 aux={'kl_divergence': 0.033935546875, 'loss': 0.0013640915276482701, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.2931095361709595}          |
| 2026-05-12 21:43:41.571 | grpo_trainer process 0 step 500] loss=0.001932514 aux={'kl_divergence': 0.047607421875, 'loss': 0.0019325140165165067, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.28637048602104187}        |
| 2026-05-12 21:55:54.464 | grpo_trainer process 0 step 510] loss=0.0032570155 aux={'kl_divergence': 0.08154296875, 'loss': 0.00325701548717916, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.34793609380722046}          |
| 2026-05-12 22:08:10.618 | grpo_trainer process 0 step 520] loss=0.0023930205 aux={'kl_divergence': 0.06005859375, 'loss': 0.002393020549789071, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.265127569437027}            |
| 2026-05-12 22:20:25.547 | grpo_trainer process 0 step 530] loss=0.008393003 aux={'kl_divergence': 0.208984375, 'loss': 0.008393002673983574, 'mean_advantage': 0.0, 'mean_reward': 0.8046875, 'std_advantage': 0.27946898341178894}             |
| 2026-05-12 22:35:24.835 | grpo_trainer process 0 step 540] loss=0.0021325625 aux={'kl_divergence': 0.05322265625, 'loss': 0.002132562454789877, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.3247135877609253}          |
| 2026-05-12 22:45:51.128 | grpo_trainer process 0 step 550] loss=0.0023343994 aux={'kl_divergence': 0.058349609375, 'loss': 0.0023343993816524744, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.2996971011161804}        |
| 2026-05-12 22:56:19.681 | grpo_trainer process 0 step 560] loss=0.0032373334 aux={'kl_divergence': 0.0810546875, 'loss': 0.003237333381548524, 'mean_advantage': 0.0, 'mean_reward': 0.85546875, 'std_advantage': 0.3247135877609253}           |
| 2026-05-12 23:06:45.136 | grpo_trainer process 0 step 570] loss=0.0031440584 aux={'kl_divergence': 0.07861328125, 'loss': 0.0031440583989024162, 'mean_advantage': 0.0, 'mean_reward': 0.85546875, 'std_advantage': 0.2996971011161804}         |
| 2026-05-12 23:17:18.412 | grpo_trainer process 0 step 580] loss=0.017267587 aux={'kl_divergence': 0.431640625, 'loss': 0.017267586663365364, 'mean_advantage': 0.0, 'mean_reward': 0.81640625, 'std_advantage': 0.31245583295822144}            |
| 2026-05-12 23:27:55.255 | grpo_trainer process 0 step 590] loss=0.007045158 aux={'kl_divergence': 0.17578125, 'loss': 0.007045158185064793, 'mean_advantage': 0.0, 'mean_reward': 0.87109375, 'std_advantage': 0.2576576769351959}              |
| 2026-05-12 23:38:44.672 | grpo_trainer process 0 step 600] loss=0.004198294 aux={'kl_divergence': 0.1064453125, 'loss': 0.004198294132947922, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.2931095361709595}              |
| 2026-05-12 23:49:29.631 | grpo_trainer process 0 step 610] loss=0.0062985504 aux={'kl_divergence': 0.158203125, 'loss': 0.00629855040460825, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.3247135877609253}             |
| 2026-05-13 00:00:04.687 | grpo_trainer process 0 step 620] loss=0.007012903 aux={'kl_divergence': 0.17578125, 'loss': 0.0070129032246768475, 'mean_advantage': 0.0, 'mean_reward': 0.81640625, 'std_advantage': 0.31245583295822144}            |
| 2026-05-13 00:10:44.568 | grpo_trainer process 0 step 630] loss=0.008049067 aux={'kl_divergence': 0.2001953125, 'loss': 0.008049067109823227, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.2996971011161804}            |
| 2026-05-13 00:22:18.352 | grpo_trainer process 0 step 640] loss=0.0066214143 aux={'kl_divergence': 0.166015625, 'loss': 0.006621414329856634, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.3422781825065613}              |
| 2026-05-13 00:33:14.139 | grpo_trainer process 0 step 650] loss=0.005589277 aux={'kl_divergence': 0.1396484375, 'loss': 0.005589277017861605, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.2931095361709595}             |
| 2026-05-13 00:44:08.552 | grpo_trainer process 0 step 660] loss=0.0088689085 aux={'kl_divergence': 0.2216796875, 'loss': 0.008868908509612083, 'mean_advantage': 0.0, 'mean_reward': 0.8046875, 'std_advantage': 0.35350340604782104}           |
| 2026-05-13 00:55:14.249 | grpo_trainer process 0 step 670] loss=0.0078261765 aux={'kl_divergence': 0.1953125, 'loss': 0.007826176472008228, 'mean_advantage': 0.0, 'mean_reward': 0.78515625, 'std_advantage': 0.33652520179748535}             |
| 2026-05-13 01:06:10.160 | grpo_trainer process 0 step 680] loss=0.00883581 aux={'kl_divergence': 0.2216796875, 'loss': 0.008835810236632824, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.3247135877609253}             |
| 2026-05-13 01:17:15.524 | grpo_trainer process 0 step 690] loss=0.009041358 aux={'kl_divergence': 0.2255859375, 'loss': 0.009041357785463333, 'mean_advantage': 0.0, 'mean_reward': 0.81640625, 'std_advantage': 0.35898441076278687}           |
| 2026-05-13 01:28:22.597 | grpo_trainer process 0 step 700] loss=0.01665382 aux={'kl_divergence': 0.416015625, 'loss': 0.016653820872306824, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.27946901321411133}               |
| 2026-05-13 01:39:26.330 | grpo_trainer process 0 step 710] loss=0.009513018 aux={'kl_divergence': 0.23828125, 'loss': 0.00951301772147417, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.3186436593532562}                |
| 2026-05-13 01:50:24.233 | grpo_trainer process 0 step 720] loss=0.010996861 aux={'kl_divergence': 0.275390625, 'loss': 0.010996861383318901, 'mean_advantage': 0.0, 'mean_reward': 0.73828125, 'std_advantage': 0.36970269680023193}            |
| 2026-05-13 02:01:35.540 | grpo_trainer process 0 step 730] loss=0.008515886 aux={'kl_divergence': 0.2119140625, 'loss': 0.008515886031091213, 'mean_advantage': 0.0, 'mean_reward': 0.76953125, 'std_advantage': 0.39025717973709106}           |
| 2026-05-13 02:12:47.008 | grpo_trainer process 0 step 740] loss=0.0068065533 aux={'kl_divergence': 0.169921875, 'loss': 0.006806553341448307, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.34793609380722046}           |
| 2026-05-13 02:23:59.507 | grpo_trainer process 0 step 750] loss=0.007927813 aux={'kl_divergence': 0.1982421875, 'loss': 0.00792781263589859, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.3061429262161255}              |
| 2026-05-13 02:35:17.269 | grpo_trainer process 0 step 760] loss=0.0072441115 aux={'kl_divergence': 0.181640625, 'loss': 0.007244111504405737, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.33652520179748535}           |
| 2026-05-13 02:46:36.622 | grpo_trainer process 0 step 770] loss=0.015708257 aux={'kl_divergence': 0.392578125, 'loss': 0.015708256512880325, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.31245583295822144}            |
| 2026-05-13 02:57:53.975 | grpo_trainer process 0 step 780] loss=0.011623598 aux={'kl_divergence': 0.291015625, 'loss': 0.011623597703874111, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.2996971011161804}             |
| 2026-05-13 03:09:18.011 | grpo_trainer process 0 step 790] loss=0.014383784 aux={'kl_divergence': 0.361328125, 'loss': 0.01438378356397152, 'mean_advantage': 0.0, 'mean_reward': 0.78125, 'std_advantage': 0.33067217469215393}                |
| 2026-05-13 03:20:37.082 | grpo_trainer process 0 step 800] loss=0.016151281 aux={'kl_divergence': 0.404296875, 'loss': 0.01615128107368946, 'mean_advantage': 0.0, 'mean_reward': 0.84375, 'std_advantage': 0.3186436593532562}                 |
| 2026-05-13 03:32:03.103 | grpo_trainer process 0 step 810] loss=0.0137720425 aux={'kl_divergence': 0.34375, 'loss': 0.013772042468190193, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.31245583295822144}               |
| 2026-05-13 03:43:28.801 | grpo_trainer process 0 step 820] loss=0.013779356 aux={'kl_divergence': 0.34375, 'loss': 0.013779356144368649, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.2723926603794098}                 |
| 2026-05-13 03:54:53.707 | grpo_trainer process 0 step 830] loss=0.010880325 aux={'kl_divergence': 0.271484375, 'loss': 0.010880324989557266, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.2996971011161804}             |
| 2026-05-13 04:06:21.311 | grpo_trainer process 0 step 840] loss=0.013759663 aux={'kl_divergence': 0.34375, 'loss': 0.013759663328528404, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.27946898341178894}                  |
| 2026-05-13 04:18:04.763 | grpo_trainer process 0 step 850] loss=0.013072835 aux={'kl_divergence': 0.328125, 'loss': 0.01307283528149128, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.2723926901817322}                 |
| 2026-05-13 04:29:41.064 | grpo_trainer process 0 step 860] loss=0.011605651 aux={'kl_divergence': 0.2890625, 'loss': 0.011605651117861271, 'mean_advantage': 0.0, 'mean_reward': 0.8125, 'std_advantage': 0.3186436593532562}                   |
| 2026-05-13 04:41:19.353 | grpo_trainer process 0 step 870] loss=0.013414731 aux={'kl_divergence': 0.3359375, 'loss': 0.013414731249213219, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.3247136175632477}               |
| 2026-05-13 04:52:57.319 | grpo_trainer process 0 step 880] loss=0.012602389 aux={'kl_divergence': 0.31640625, 'loss': 0.012602388858795166, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.33652520179748535}             |
| 2026-05-13 05:04:39.931 | grpo_trainer process 0 step 890] loss=0.022064399 aux={'kl_divergence': 0.55078125, 'loss': 0.02206439897418022, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.34793609380722046}              |
| 2026-05-13 05:16:21.618 | grpo_trainer process 0 step 900] loss=0.015547693 aux={'kl_divergence': 0.38671875, 'loss': 0.015547692775726318, 'mean_advantage': 0.0, 'mean_reward': 0.84375, 'std_advantage': 0.3186436593532562}                 |
| 2026-05-13 05:28:07.983 | grpo_trainer process 0 step 910] loss=0.01896976 aux={'kl_divergence': 0.474609375, 'loss': 0.018969759345054626, 'mean_advantage': 0.0, 'mean_reward': 0.78515625, 'std_advantage': 0.36970269680023193}             |
| 2026-05-13 05:39:58.929 | grpo_trainer process 0 step 920] loss=0.017098835 aux={'kl_divergence': 0.427734375, 'loss': 0.017098834738135338, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.33652520179748535}            |
| 2026-05-13 05:51:56.950 | grpo_trainer process 0 step 930] loss=0.015420125 aux={'kl_divergence': 0.38671875, 'loss': 0.015420124866068363, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.33652520179748535}             |
| 2026-05-13 06:04:57.645 | grpo_trainer process 0 step 940] loss=0.027911901 aux={'kl_divergence': 0.69921875, 'loss': 0.027911901473999023, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.33067217469215393}               |
| 2026-05-13 06:17:00.508 | grpo_trainer process 0 step 950] loss=0.03748191 aux={'kl_divergence': 0.9375, 'loss': 0.03748191148042679, 'mean_advantage': 0.0, 'mean_reward': 0.84765625, 'std_advantage': 0.28637048602104187}                   |
| 2026-05-13 06:29:03.555 | grpo_trainer process 0 step 960] loss=0.05241655 aux={'kl_divergence': 1.3046875, 'loss': 0.05241655185818672, 'mean_advantage': 0.0, 'mean_reward': 0.8046875, 'std_advantage': 0.33067217469215393}                 |
| 2026-05-13 06:41:11.533 | grpo_trainer process 0 step 970] loss=0.064355195 aux={'kl_divergence': 1.609375, 'loss': 0.06435519456863403, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.2931095361709595}                  |
| 2026-05-13 06:53:26.454 | grpo_trainer process 0 step 980] loss=0.08124774 aux={'kl_divergence': 2.03125, 'loss': 0.0812477394938469, 'mean_advantage': 0.0, 'mean_reward': 0.875, 'std_advantage': 0.3061429262161255}                         |
| 2026-05-13 07:05:37.114 | grpo_trainer process 0 step 990] loss=0.08639534 aux={'kl_divergence': 2.15625, 'loss': 0.08639533817768097, 'mean_advantage': 0.0, 'mean_reward': 0.84765625, 'std_advantage': 0.31245583295822144}                  |
| 2026-05-13 07:17:45.600 | grpo_trainer process 0 step 1000] loss=0.05171673 aux={'kl_divergence': 1.2890625, 'loss': 0.05171672999858856, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.3061429262161255}                 |
