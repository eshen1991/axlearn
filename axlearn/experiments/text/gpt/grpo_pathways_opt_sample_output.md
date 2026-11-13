## Job Launch Command
```shell
source ./axlearn/tools/setup_grpo_env.sh
export NAME=eshen-v6e-rl-grpo-pathways-opt
export OUTPUT_DIR="gs://<gcs-bucket>/users/ericshen/${NAME}/1794441712"
export MLDIAGNOSTICS_RUN_GROUP="eshen-grpo-experiment-1794441712"

➜  ~ nohup axlearn gcp launch run \
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
  --pathways_head_mem=128 \
  --pathways_head_cpu=8 \
  --env=GRPO_REWARD_TYPE:gsm8k \
  -- \
  python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.grpo_native_example \
  --config=$CONFIG \
  --trainer_dir=$OUTPUT_DIR \
  --data_dir=gs://<gcs-bucket>/tensorflow-datasets \
  --jax_backend=proxy \
  --use_mldiagnostics=True \
  --mldiagnostics_run_group=$MLDIAGNOSTICS_RUN_GROUP \
  --mldiagnostics_run_name=$NAME \
  --trainer_log_every_n_steps=10 \
  --trace_at_steps=5,20,40,80 \
  --n_steps_for_each_trace=5 > axlearn_launcher.log 2>&1 &

```
## output directory gs://<gcs-bucket>/users/ericshen/eshen-v6e-rl-grpo-pathways-opt/1794441712
## trainer_config
```shell
➜  ~ gcloud storage cat gs://<gcs-bucket>/users/ericshen/eshen-v6e-rl-grpo-pathways-opt/1794441712/trainer_config
batch_axis_names: 'data'
checkpointer.gc_loop_interval_seconds: 60
checkpointer.keep_last_n: 2
checkpointer.klass: 'axlearn.common.checkpointer.Checkpointer'
checkpointer.save_policy.fn: 'axlearn.common.checkpointer.every_n_steps_policy'
checkpointer.save_policy.min_step: 1
checkpointer.save_policy.n: 100
checkpointer.storage.klass: 'axlearn.common.checkpointer.TensorStoreStateStorage'
checkpointer.storage.timeout_secs: 3600
crash_on_hang_timeout_seconds: 7200
dir: 'gs://<gcs-bucket>/users/ericshen/eshen-v6e-rl-grpo-pathways-opt/1794441712'
enable_async_rollout: True
init_state_builder.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOStateBuilder'
init_state_builder.storage_builder.concurrent_gb: 4
init_state_builder.storage_builder.dir: 'gs://<gcs-bucket>/checkpoints/llama-3-1-8B-instruct/step_00000000'
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
learner.update_rules[0][0]: 'reference/.*'
learner.update_rules[0][1]: <UpdateType.NO_UPDATE: 'no_update'>
learner.update_rules[1][0]: 'sampler/.*'
learner.update_rules[1][1]: <UpdateType.NO_UPDATE: 'no_update'>
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
mldiagnostics_environment: 'prod'
mldiagnostics_run_group: 'eshen-grpo-experiment-1794441712'
mldiagnostics_run_name: 'eshen-v6e-rl-grpo-pathways-opt'
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
model.actor.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: None
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
model.reference.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: None
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
model.sampler.batch_axis_names: None
model.sampler.decoder.attention_mask: None
model.sampler.decoder.decoding.klass: 'axlearn.common.decoder.DecodingLayer'
model.sampler.decoder.dim: 4096
model.sampler.decoder.dropout_rate: 0.0
model.sampler.decoder.dtype: 'jax.numpy.bfloat16'
model.sampler.decoder.emb.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.emb.klass: 'axlearn.common.embedding.TransformerTextEmbeddings'
model.sampler.decoder.emb.token_emb.klass: 'axlearn.common.layers.Embedding'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].fan: 'fan_out'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.sampler.decoder.emb.token_emb.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.sampler.decoder.emb.token_emb.param_partition_spec[0]: 'model'
model.sampler.decoder.emb.token_emb.param_partition_spec[1][0]: 'expert'
model.sampler.decoder.emb.token_emb.param_partition_spec[1][1]: 'fsdp'
model.sampler.decoder.emb.token_emb.param_partition_spec[1][2]: 'seq'
model.sampler.decoder.eos_token_id: 128001
model.sampler.decoder.klass: 'axlearn.common.decoder.Decoder'
model.sampler.decoder.lm_head.klass: 'axlearn.common.decoder.LmHead'
model.sampler.decoder.lm_head.param_partition_spec[0]: 'model'
model.sampler.decoder.lm_head.param_partition_spec[1][0]: 'expert'
model.sampler.decoder.lm_head.param_partition_spec[1][1]: 'fsdp'
model.sampler.decoder.lm_head.param_partition_spec[1][2]: 'seq'
model.sampler.decoder.logits_partition_spec[0][0]: 'data'
model.sampler.decoder.logits_partition_spec[0][1]: 'expert'
model.sampler.decoder.logits_partition_spec[0][2]: 'fsdp'
model.sampler.decoder.logits_partition_spec[1]: 'seq'
model.sampler.decoder.logits_partition_spec[2]: 'model'
model.sampler.decoder.output_dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.output_norm.eps: 1e-05
model.sampler.decoder.output_norm.forward_dtype: None
model.sampler.decoder.output_norm.klass: 'axlearn.common.layers.RMSNorm'
model.sampler.decoder.pad_token_id: 128004
model.sampler.decoder.transformer.klass: 'axlearn.common.attention.RepeatedTransformerLayer'
model.sampler.decoder.transformer.layer.feed_forward.activation[0]: 'nn.silu'
model.sampler.decoder.transformer.layer.feed_forward.activation[1]: 'linear'
model.sampler.decoder.transformer.layer.feed_forward.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.transformer.layer.feed_forward.hidden_dim.fn: 'axlearn.experiments.text.gpt.common.scale_fn'
model.sampler.decoder.transformer.layer.feed_forward.hidden_dim.round_up_to_multiples_of: 256
model.sampler.decoder.transformer.layer.feed_forward.hidden_dim.scale: 3.5
model.sampler.decoder.transformer.layer.feed_forward.klass: 'axlearn.common.attention.TransformerFeedForwardLayer'
model.sampler.decoder.transformer.layer.feed_forward.linear1.bias: False
model.sampler.decoder.transformer.layer.feed_forward.linear1.klass: 'axlearn.common.layers.Linear'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][0]: 'data'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][1]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][2]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[1]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[2]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][0]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][1]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][2]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[1]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear2.bias: False
model.sampler.decoder.transformer.layer.feed_forward.linear2.klass: 'axlearn.common.layers.Linear'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][0]: 'data'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][1]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][2]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[1]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[2]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[0]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][0]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][1]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][2]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.norm.eps: 1e-05
model.sampler.decoder.transformer.layer.feed_forward.norm.forward_dtype: None
model.sampler.decoder.transformer.layer.feed_forward.norm.klass: 'axlearn.common.layers.RMSNorm'
model.sampler.decoder.transformer.layer.feed_forward.residual_weight: 1.0
model.sampler.decoder.transformer.layer.feed_forward.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.sampler.decoder.transformer.layer.feed_forward.stochastic_depth.mode: 'row'
model.sampler.decoder.transformer.layer.feed_forward.structure: 'prenorm'
model.sampler.decoder.transformer.layer.klass: 'axlearn.common.attention.TransformerLayer'
model.sampler.decoder.transformer.layer.remat_spec['prevent_cse']: False
model.sampler.decoder.transformer.layer.remat_spec['policy'].fn: 'axlearn.common.attention._save_and_offload_only_these_names_regex'
model.sampler.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_offloaded: None
model.sampler.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: None
model.sampler.decoder.transformer.layer.remat_spec['policy'].offload_dst: 'pinned_host'
model.sampler.decoder.transformer.layer.remat_spec['policy'].offload_src: 'device'
model.sampler.decoder.transformer.layer.self_attention.attention.causal: True
model.sampler.decoder.transformer.layer.self_attention.attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.klass: 'axlearn.common.attention.FusedGroupedQKVLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.bias: False
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.klass: 'axlearn.common.attention.MultiheadInputLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][0]: 'expert'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][1]: 'fsdp'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][2]: 'seq'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[1]: 'model'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[2]: None
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.num_kv_heads: 8
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.klass: 'axlearn.common.attention.RoFormerQKVLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.klass: 'axlearn.common.attention.RoFormerSinusoidalPositionalEmbedding'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.theta: 500000.0
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.rotary_value: False
model.sampler.decoder.transformer.layer.self_attention.attention.key_scale.klass: 'axlearn.common.attention.ScaleKey'
model.sampler.decoder.transformer.layer.self_attention.attention.klass: 'axlearn.common.attention.GroupedQueryAttention'
model.sampler.decoder.transformer.layer.self_attention.attention.kv_cache.cache_dtype: 'jax.numpy.bfloat16'
model.sampler.decoder.transformer.layer.self_attention.attention.kv_cache.klass: 'axlearn.common.kv_cache.kv_cache.KVCache'
model.sampler.decoder.transformer.layer.self_attention.attention.num_heads: 32
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.bias: False
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.klass: 'axlearn.common.attention.MultiheadOutputLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][0]: 'expert'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][1]: 'fsdp'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][2]: 'seq'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[1]: 'model'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[2]: None
model.sampler.decoder.transformer.layer.self_attention.attention.query_scale.klass: 'axlearn.common.attention.ScaleQuery'
model.sampler.decoder.transformer.layer.self_attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.transformer.layer.self_attention.klass: 'axlearn.common.attention.TransformerAttentionLayer'
model.sampler.decoder.transformer.layer.self_attention.norm.eps: 1e-05
model.sampler.decoder.transformer.layer.self_attention.norm.forward_dtype: None
model.sampler.decoder.transformer.layer.self_attention.norm.klass: 'axlearn.common.layers.RMSNorm'
model.sampler.decoder.transformer.layer.self_attention.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.sampler.decoder.transformer.layer.self_attention.stochastic_depth.mode: 'row'
model.sampler.decoder.transformer.layer.self_attention.structure: 'prenorm'
model.sampler.decoder.transformer.num_layers: 32
model.sampler.decoder.transformer.repeat.drop_output.fn: 'axlearn.common.repeat._drop_by_regex'
model.sampler.decoder.transformer.repeat.drop_output.rules[0]: 'module_outputs.*'
model.sampler.decoder.transformer.repeat.klass: 'axlearn.common.attention._TransformerRepeat'
model.sampler.decoder.vocab_size: 128256
model.sampler.dtype: 'jax.numpy.bfloat16'
model.sampler.klass: 'axlearn.common.causal_lm.Model'
model.sampler.metrics.klass: 'axlearn.common.causal_lm.CompositeLossMetrics'
model.sampler.metrics.metrics['lm'].klass: 'axlearn.common.causal_lm.CrossEntropyLossMetrics'
model.sampler.metrics.metrics['aux'].klass: 'axlearn.common.causal_lm.AuxLossMetrics'
model.sampler.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.sampler.param_init.init_by_param_name['.*weight$'].fan: 'fan_in'
model.sampler.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.sampler.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.sampler.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
n_steps_for_each_trace: 5
name: 'grpo_trainer'
num_generations: 2
prune_empty_state_updates: True
recorder.fn: '__main__.<lambda>'
reward_type: 'gsm8k'
rollout_tp_degree: 4
save_input_iterator: False
start_trace_process_indices[0]: 0
start_trace_steps[0]: 5
start_trace_steps[1]: 20
start_trace_steps[2]: 40
start_trace_steps[3]: 80
summary_writer.klass: 'axlearn.common.summary_writer.SummaryWriter'
summary_writer.max_queue: 1000
summary_writer.write_every_n_steps: 10
use_mldiagnostics: True
vocab.filename: 'Llama-3-tokenizer.json'
vocab.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOV3Vocabulary'
watchdog_timeout_seconds: 3600
```

## model_analysis.txt
```shell
➜  ~ gcloud storage cat gs://<gcs-bucket>/users/ericshen/eshen-v6e-rl-grpo-pathways-opt/1794441712/model_analysis.txt | tr '\t' '\n'
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
State: learner/optimizer/2/count=int32(()) mesh_axes=TensorSpec(shape=[], dtype=<class 'jax.numpy.int32'>, mesh_axes=PartitionSpec(), memory_kind=None)
Training state size: 59.83 GiB
Training state size (partitioned): 7.48 GiB
Max training state size (partitioned): 7.48 GiB
##########################################################

```

## Training Metrics

### Definition
---

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
*   **Intuition**: This measures the Actor's "deviation penalty." We want this to remain stable and small (e.g. `0.005 - 0.05`). If it surges too high, the model is "forgetting" its pre-trained knowledge or collapsing into gibberish to exploit the rewards.

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
*   **Intuition**: This is the **absolute scoreboard** showing how smart your model is in the real world. A value of `0.914` means exactly **91.4% of all math questions solved in this batch were completely correct.**

---

#### 5. **`std_advantage`** (The Learning Signal Contrast)
*   **Definition**: The standard deviation (the spread/variance) of the relative advantages. It measures the **strength and presence of contrast** inside your training groups.
*   **Plain-Text Formula**:
    `Std_Advantage = Standard_Deviation(Advantages_in_Batch)`
*   **Intuition**: Think of this as the **learning signal strength**.
    *   If it is positive ($>0.0$): It confirms that in some groups, some siblings were correct and others were incorrect. This **contrast** provides a strong active gradient signal to the Actor model.
    *   If it were $0.0$: All siblings got the exact same reward (no contrast), and the model would learn nothing.



### Step Time Calculation
To evaluate end-to-end performance across Steps 1 to 1000 on a Spot `tpu-v6e-16` slice (`4x4`), we compare both total wall-clock elapsed time (including GCE Spot preemption/re-provisioning queue waits) and active TPU training time.

#### Overall Average Calculation

The training run logged Step 1 at **2026-11-12 00:06:38 UTC** and reached Step 1000 at **2026-11-12 18:48:16 UTC** (`Checkpointer flushed` at `18:49:48 UTC`).

*   **Total Wall-Clock Elapsed Time (incl. 8 Spot TPU preemption queue waits):** 18 hours, 41 minutes, and 38 seconds ($67,298.52$ seconds; vs. 2 days, 10 hours, 55 minutes in baseline).
*   **Active TPU Training Time (excl. Spot queue waits):** **2 hours, 52 minutes, and 39 seconds** ($10,359.47$ seconds across $919$ uninterrupted steps, including all 4 Managed XProf trace uploads and periodic GCS checkpoints).
*   **Uninterrupted Steps 1–500 (`every_n_steps_policy(n=100)`):** **1 hour, 27 minutes, and 58 seconds** ($5,278.30$ seconds for $499$ steps = **$10.58$ seconds/step**).
*   **Pure Steady-State Step Time (untraced, non-checkpoint intervals):** **$9.56$ seconds/step** ($9.49$ seconds/step inside the Step 80 Managed XProf trace; **$15.2\times$ faster** than the $145.29$s baseline).

#### Comparison to Baseline (`1779314896` / `1782796849`)

| Metric | Baseline (Seconds) | Optimized (Seconds) | Speedup |
| :--- | :---: | :---: | :---: |
| **Pure Steady-State Step Time** | 145.29 | **9.56** | **15.2x** |
| **Active Training Average (`n=100`, Steps 1–500)** | 206.94 | **10.58** | **19.6x** |
| **Active Training Average (All 1000 Steps, incl. `n=25` Spot resilience)** | 206.94 | **11.27** | **18.4x** |
| **Wall-Clock Average (incl. GCE Spot queue waits)** | 212.32 | **67.37** | **3.15x** |

During Steps 501–1000, `us-central1-b` experienced heavy Spot churn (~20-minute slice lifetimes with 30–250 minute GCE Spot allocation waits), so checkpoint frequency was tightened from `n=100` to `n=25` (~5.5 minutes) to lock in progress across 8 Spot preemptions. Excluding those GCE Spot queue waits, active step time remained rock-solid at **$10.58$–$11.27$ seconds/step** throughout all 1,000 steps while GSM8K `mean_reward` climbed from **$68.0\%$** at Step 1 to a peak of **$91.4\%$** at Step 820 (**$88.0\%$** average across Steps 910–1000, `kl_divergence = 0.016`).

### Metrics Log
| timestamp | message |
| :--- | :--- |
| 2026-11-12T00:06:38.011635000Z | grpo_trainer process   0 step        1] loss=0.0012929136 aux={'kl_divergence': 0.0, 'loss': 0.0012929135700687766, 'mean_advantage': 0.0, 'mean_reward': 0.6796875, 'std_advantage': 0.4049890339374542} |
| 2026-11-12T00:06:45.556228000Z | grpo_trainer process   0 step        2] loss=4564.9062 aux={'kl_divergence': 114176.0, 'loss': 4564.90625, 'mean_advantage': 0.0, 'mean_reward': 0.6953125, 'std_advantage': 0.3643829822540283} |
| 2026-11-12T00:06:55.095967000Z | grpo_trainer process   0 step        3] loss=0.5119411 aux={'kl_divergence': 12.8125, 'loss': 0.5119410753250122, 'mean_advantage': 0.0, 'mean_reward': 0.8046875, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T00:07:04.760131000Z | grpo_trainer process   0 step        4] loss=0.0016620684 aux={'kl_divergence': 0.0322265625, 'loss': 0.0016620684182271361, 'mean_advantage': 0.0, 'mean_reward': 0.6640625, 'std_advantage': 0.3643829822540283} |
| 2026-11-12T00:07:14.345730000Z | grpo_trainer process   0 step        5] loss=0.0026647632 aux={'kl_divergence': 0.0634765625, 'loss': 0.0026647632475942373, 'mean_advantage': 0.0, 'mean_reward': 0.60546875, 'std_advantage': 0.39025720953941345} |
| 2026-11-12T00:08:02.522489000Z | grpo_trainer process   0 step       10] loss=0.0016385185 aux={'kl_divergence': 0.03955078125, 'loss': 0.0016385185299441218, 'mean_advantage': 0.0, 'mean_reward': 0.37109375, 'std_advantage': 0.43743813037872314} |
| 2026-11-12T00:09:57.776330000Z | grpo_trainer process   0 step       20] loss=0.0020785816 aux={'kl_divergence': 0.051025390625, 'loss': 0.002078581601381302, 'mean_advantage': 0.0, 'mean_reward': 0.1953125, 'std_advantage': 0.40498900413513184} |
| 2026-11-12T00:11:50.367753000Z | grpo_trainer process   0 step       30] loss=0.0019821832 aux={'kl_divergence': 0.05029296875, 'loss': 0.0019821831956505775, 'mean_advantage': 0.0, 'mean_reward': 0.16015625, 'std_advantage': 0.36970269680023193} |
| 2026-11-12T00:13:26.447342000Z | grpo_trainer process   0 step       40] loss=0.002149192 aux={'kl_divergence': 0.053955078125, 'loss': 0.0021491919178515673, 'mean_advantage': 0.0, 'mean_reward': 0.15234375, 'std_advantage': 0.33652520179748535} |
| 2026-11-12T00:15:17.369211000Z | grpo_trainer process   0 step       50] loss=0.0022274477 aux={'kl_divergence': 0.055419921875, 'loss': 0.0022274476941674948, 'mean_advantage': 0.0, 'mean_reward': 0.171875, 'std_advantage': 0.3643829822540283} |
| 2026-11-12T00:16:51.651771000Z | grpo_trainer process   0 step       60] loss=0.0023793103 aux={'kl_divergence': 0.059814453125, 'loss': 0.002379310317337513, 'mean_advantage': 0.0, 'mean_reward': 0.18359375, 'std_advantage': 0.33652520179748535} |
| 2026-11-12T00:18:27.593979000Z | grpo_trainer process   0 step       70] loss=0.0021753167 aux={'kl_divergence': 0.05224609375, 'loss': 0.0021753166802227497, 'mean_advantage': 0.0, 'mean_reward': 0.16015625, 'std_advantage': 0.3801189064979553} |
| 2026-11-12T00:20:03.655423000Z | grpo_trainer process   0 step       80] loss=0.0023164072 aux={'kl_divergence': 0.0576171875, 'loss': 0.002316407160833478, 'mean_advantage': 0.0, 'mean_reward': 0.14453125, 'std_advantage': 0.35898441076278687} |
| 2026-11-12T00:21:57.570218000Z | grpo_trainer process   0 step       90] loss=0.0023599442 aux={'kl_divergence': 0.058837890625, 'loss': 0.002359944162890315, 'mean_advantage': 0.0, 'mean_reward': 0.16015625, 'std_advantage': 0.36970269680023193} |
| 2026-11-12T00:23:33.597902000Z | grpo_trainer process   0 step      100] loss=0.0027205471 aux={'kl_divergence': 0.06787109375, 'loss': 0.002720547141507268, 'mean_advantage': 0.0, 'mean_reward': 0.2109375, 'std_advantage': 0.3749469816684723} |
| 2026-11-12T00:27:08.347001000Z | grpo_trainer process   0 step      110] loss=0.002115872 aux={'kl_divergence': 0.05322265625, 'loss': 0.0021158719900995493, 'mean_advantage': 0.0, 'mean_reward': 0.15234375, 'std_advantage': 0.34793609380722046} |
| 2026-11-12T00:28:43.029022000Z | grpo_trainer process   0 step      120] loss=0.002212918 aux={'kl_divergence': 0.05517578125, 'loss': 0.002212917897850275, 'mean_advantage': 0.0, 'mean_reward': 0.1796875, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T00:30:19.014786000Z | grpo_trainer process   0 step      130] loss=0.0022982056 aux={'kl_divergence': 0.056640625, 'loss': 0.0022982056252658367, 'mean_advantage': 0.0, 'mean_reward': 0.16796875, 'std_advantage': 0.3589843809604645} |
| 2026-11-12T00:31:55.226345000Z | grpo_trainer process   0 step      140] loss=0.0023056935 aux={'kl_divergence': 0.056884765625, 'loss': 0.002305693458765745, 'mean_advantage': 0.0, 'mean_reward': 0.22265625, 'std_advantage': 0.3801189064979553} |
| 2026-11-12T00:33:31.757780000Z | grpo_trainer process   0 step      150] loss=0.0023307137 aux={'kl_divergence': 0.0576171875, 'loss': 0.0023307136725634336, 'mean_advantage': 0.0, 'mean_reward': 0.1640625, 'std_advantage': 0.3952288329601288} |
| 2026-11-12T00:35:08.003255000Z | grpo_trainer process   0 step      160] loss=0.0023382439 aux={'kl_divergence': 0.058837890625, 'loss': 0.002338243881240487, 'mean_advantage': 0.0, 'mean_reward': 0.171875, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T00:36:43.669435000Z | grpo_trainer process   0 step      170] loss=0.0022957285 aux={'kl_divergence': 0.056640625, 'loss': 0.002295728540048003, 'mean_advantage': 0.0, 'mean_reward': 0.171875, 'std_advantage': 0.3852214515209198} |
| 2026-11-12T00:38:18.571058000Z | grpo_trainer process   0 step      180] loss=0.0022007457 aux={'kl_divergence': 0.0556640625, 'loss': 0.0022007457446306944, 'mean_advantage': 0.0, 'mean_reward': 0.19140625, 'std_advantage': 0.4001387059688568} |
| 2026-11-12T00:39:54.535465000Z | grpo_trainer process   0 step      190] loss=0.0022095253 aux={'kl_divergence': 0.0556640625, 'loss': 0.0022095253225415945, 'mean_advantage': 0.0, 'mean_reward': 0.19921875, 'std_advantage': 0.3801189064979553} |
| 2026-11-12T00:41:30.976078000Z | grpo_trainer process   0 step      200] loss=0.0022317774 aux={'kl_divergence': 0.055419921875, 'loss': 0.002231777412816882, 'mean_advantage': 0.0, 'mean_reward': 0.2578125, 'std_advantage': 0.4329514801502228} |
| 2026-11-12T00:44:53.137531000Z | grpo_trainer process   0 step      210] loss=0.0022403987 aux={'kl_divergence': 0.055419921875, 'loss': 0.0022403986658900976, 'mean_advantage': 0.0, 'mean_reward': 0.23828125, 'std_advantage': 0.36970269680023193} |
| 2026-11-12T00:46:29.366118000Z | grpo_trainer process   0 step      220] loss=0.0020805122 aux={'kl_divergence': 0.053466796875, 'loss': 0.0020805122330784798, 'mean_advantage': 0.0, 'mean_reward': 0.26953125, 'std_advantage': 0.43743813037872314} |
| 2026-11-12T00:48:04.294854000Z | grpo_trainer process   0 step      230] loss=0.0022372333 aux={'kl_divergence': 0.0556640625, 'loss': 0.0022372333332896233, 'mean_advantage': 0.0, 'mean_reward': 0.26171875, 'std_advantage': 0.428417831659317} |
| 2026-11-12T00:49:39.667822000Z | grpo_trainer process   0 step      240] loss=0.0020914434 aux={'kl_divergence': 0.0537109375, 'loss': 0.0020914433989673853, 'mean_advantage': 0.0, 'mean_reward': 0.31640625, 'std_advantage': 0.43743813037872314} |
| 2026-11-12T00:51:15.783905000Z | grpo_trainer process   0 step      250] loss=0.0019548137 aux={'kl_divergence': 0.050048828125, 'loss': 0.0019548137206584215, 'mean_advantage': 0.0, 'mean_reward': 0.29296875, 'std_advantage': 0.43743813037872314} |
| 2026-11-12T00:52:52.111566000Z | grpo_trainer process   0 step      260] loss=0.0022328792 aux={'kl_divergence': 0.053955078125, 'loss': 0.0022328791674226522, 'mean_advantage': 0.0, 'mean_reward': 0.3671875, 'std_advantage': 0.4329514801502228} |
| 2026-11-12T00:54:28.271604000Z | grpo_trainer process   0 step      270] loss=0.0018844312 aux={'kl_divergence': 0.04736328125, 'loss': 0.0018844312289729714, 'mean_advantage': 0.0, 'mean_reward': 0.48046875, 'std_advantage': 0.4462761878967285} |
| 2026-11-12T00:56:04.219373000Z | grpo_trainer process   0 step      280] loss=0.0017318164 aux={'kl_divergence': 0.04296875, 'loss': 0.0017318164464086294, 'mean_advantage': 0.0, 'mean_reward': 0.4921875, 'std_advantage': 0.4145194888114929} |
| 2026-11-12T00:57:38.752326000Z | grpo_trainer process   0 step      290] loss=0.0015940206 aux={'kl_divergence': 0.040283203125, 'loss': 0.0015940206358209252, 'mean_advantage': 0.0, 'mean_reward': 0.49609375, 'std_advantage': 0.4192034602165222} |
| 2026-11-12T00:59:14.267071000Z | grpo_trainer process   0 step      300] loss=0.0014312628 aux={'kl_divergence': 0.03515625, 'loss': 0.001431262819096446, 'mean_advantage': 0.0, 'mean_reward': 0.55859375, 'std_advantage': 0.4800037145614624} |
| 2026-11-12T01:02:32.895917000Z | grpo_trainer process   0 step      310] loss=0.0015091421 aux={'kl_divergence': 0.03564453125, 'loss': 0.0015091421082615852, 'mean_advantage': 0.0, 'mean_reward': 0.671875, 'std_advantage': 0.45921438932418823} |
| 2026-11-12T01:04:09.339613000Z | grpo_trainer process   0 step      320] loss=0.0013411484 aux={'kl_divergence': 0.033935546875, 'loss': 0.0013411483960226178, 'mean_advantage': 0.0, 'mean_reward': 0.58203125, 'std_advantage': 0.4462761878967285} |
| 2026-11-12T01:05:45.246401000Z | grpo_trainer process   0 step      330] loss=0.0013506428 aux={'kl_divergence': 0.03271484375, 'loss': 0.0013506427640095353, 'mean_advantage': 0.0, 'mean_reward': 0.57421875, 'std_advantage': 0.4960082471370697} |
| 2026-11-12T01:07:21.280691000Z | grpo_trainer process   0 step      340] loss=0.001434225 aux={'kl_divergence': 0.03515625, 'loss': 0.0014342250069603324, 'mean_advantage': 0.0, 'mean_reward': 0.61328125, 'std_advantage': 0.46344685554504395} |
| 2026-11-12T01:08:55.398071000Z | grpo_trainer process   0 step      350] loss=0.0012376715 aux={'kl_divergence': 0.03173828125, 'loss': 0.0012376714730635285, 'mean_advantage': 0.0, 'mean_reward': 0.64453125, 'std_advantage': 0.41920343041419983} |
| 2026-11-12T01:10:31.365734000Z | grpo_trainer process   0 step      360] loss=0.0011159673 aux={'kl_divergence': 0.030517578125, 'loss': 0.0011159672867506742, 'mean_advantage': 0.0, 'mean_reward': 0.68359375, 'std_advantage': 0.43743813037872314} |
| 2026-11-12T01:12:07.423872000Z | grpo_trainer process   0 step      370] loss=0.0020354684 aux={'kl_divergence': 0.0478515625, 'loss': 0.002035468351095915, 'mean_advantage': 0.0, 'mean_reward': 0.67578125, 'std_advantage': 0.4192034602165222} |
| 2026-11-12T01:13:43.943355000Z | grpo_trainer process   0 step      380] loss=0.0012695786 aux={'kl_divergence': 0.0294189453125, 'loss': 0.0012695785844698548, 'mean_advantage': 0.0, 'mean_reward': 0.67578125, 'std_advantage': 0.43743813037872314} |
| 2026-11-12T01:15:20.097339000Z | grpo_trainer process   0 step      390] loss=0.0014334519 aux={'kl_divergence': 0.03466796875, 'loss': 0.0014334518928080797, 'mean_advantage': 0.0, 'mean_reward': 0.71484375, 'std_advantage': 0.39025717973709106} |
| 2026-11-12T01:16:56.280942000Z | grpo_trainer process   0 step      400] loss=0.0011039521 aux={'kl_divergence': 0.0279541015625, 'loss': 0.0011039520613849163, 'mean_advantage': 0.0, 'mean_reward': 0.65625, 'std_advantage': 0.3952288329601288} |
| 2026-11-12T01:20:13.615216000Z | grpo_trainer process   0 step      410] loss=0.0010862 aux={'kl_divergence': 0.02783203125, 'loss': 0.0010862000053748488, 'mean_advantage': 0.0, 'mean_reward': 0.7265625, 'std_advantage': 0.4238356947898865} |
| 2026-11-12T01:21:49.704817000Z | grpo_trainer process   0 step      420] loss=0.0009860062 aux={'kl_divergence': 0.0252685546875, 'loss': 0.0009860062273219228, 'mean_advantage': 0.0, 'mean_reward': 0.734375, 'std_advantage': 0.3952288031578064} |
| 2026-11-12T01:23:25.669725000Z | grpo_trainer process   0 step      430] loss=0.0009221301 aux={'kl_divergence': 0.024169921875, 'loss': 0.0009221300715580583, 'mean_advantage': 0.0, 'mean_reward': 0.72265625, 'std_advantage': 0.4001387059688568} |
| 2026-11-12T01:25:02.034941000Z | grpo_trainer process   0 step      440] loss=0.00088872697 aux={'kl_divergence': 0.0224609375, 'loss': 0.0008887269650585949, 'mean_advantage': 0.0, 'mean_reward': 0.7578125, 'std_advantage': 0.4049890339374542} |
| 2026-11-12T01:26:38.060992000Z | grpo_trainer process   0 step      450] loss=0.00081709283 aux={'kl_divergence': 0.0201416015625, 'loss': 0.0008170928340405226, 'mean_advantage': 0.0, 'mean_reward': 0.796875, 'std_advantage': 0.35350340604782104} |
| 2026-11-12T01:28:13.624691000Z | grpo_trainer process   0 step      460] loss=0.00070863485 aux={'kl_divergence': 0.017333984375, 'loss': 0.0007086348487064242, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.2723926603794098} |
| 2026-11-12T01:29:48.272962000Z | grpo_trainer process   0 step      470] loss=0.0005817547 aux={'kl_divergence': 0.0155029296875, 'loss': 0.0005817547207698226, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.31245583295822144} |
| 2026-11-12T01:31:24.087826000Z | grpo_trainer process   0 step      480] loss=0.00048102153 aux={'kl_divergence': 0.0128173828125, 'loss': 0.00048102153232321143, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T01:33:00.312514000Z | grpo_trainer process   0 step      490] loss=0.0004332917 aux={'kl_divergence': 0.01104736328125, 'loss': 0.000433291686931625, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.28637048602104187} |
| 2026-11-12T01:34:36.309788000Z | grpo_trainer process   0 step      500] loss=0.00033097237 aux={'kl_divergence': 0.0081787109375, 'loss': 0.0003309723688289523, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.2723926603794098} |
| 2026-11-12T05:54:16.938573000Z | grpo_trainer process   0 step      510] loss=0.00036332308 aux={'kl_divergence': 0.00823974609375, 'loss': 0.00036332308081910014, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.3061429262161255} |
| 2026-11-12T05:55:52.233051000Z | grpo_trainer process   0 step      520] loss=0.0003132224 aux={'kl_divergence': 0.008056640625, 'loss': 0.00031322240829467773, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.28637048602104187} |
| 2026-11-12T05:58:56.480848000Z | grpo_trainer process   0 step      530] loss=0.0004467201 aux={'kl_divergence': 0.010009765625, 'loss': 0.00044672010699287057, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.2931095361709595} |
| 2026-11-12T06:00:31.846373000Z | grpo_trainer process   0 step      540] loss=0.0004028622 aux={'kl_divergence': 0.00927734375, 'loss': 0.00040286220610141754, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.27946901321411133} |
| 2026-11-12T06:02:07.307597000Z | grpo_trainer process   0 step      550] loss=0.00032227888 aux={'kl_divergence': 0.00848388671875, 'loss': 0.0003222788800485432, 'mean_advantage': 0.0, 'mean_reward': 0.875, 'std_advantage': 0.27946901321411133} |
| 2026-11-12T06:05:04.519118000Z | grpo_trainer process   0 step      560] loss=0.00035570952 aux={'kl_divergence': 0.0081787109375, 'loss': 0.0003557095187716186, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.265127569437027} |
| 2026-11-12T06:06:39.807604000Z | grpo_trainer process   0 step      570] loss=0.0003991748 aux={'kl_divergence': 0.010009765625, 'loss': 0.0003991748089902103, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T06:37:55.500853000Z | grpo_trainer process   0 step      580] loss=0.0003094128 aux={'kl_divergence': 0.0084228515625, 'loss': 0.00030941280419938266, 'mean_advantage': 0.0, 'mean_reward': 0.88671875, 'std_advantage': 0.24202723801136017} |
| 2026-11-12T06:39:30.471036000Z | grpo_trainer process   0 step      590] loss=0.00029946901 aux={'kl_divergence': 0.00799560546875, 'loss': 0.0002994690148625523, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.2931095361709595} |
| 2026-11-12T06:41:05.655215000Z | grpo_trainer process   0 step      600] loss=0.00032713034 aux={'kl_divergence': 0.008544921875, 'loss': 0.00032713034306652844, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.23382051289081573} |
| 2026-11-12T06:44:16.053806000Z | grpo_trainer process   0 step      610] loss=0.0004008633 aux={'kl_divergence': 0.00872802734375, 'loss': 0.00040086329681798816, 'mean_advantage': 0.0, 'mean_reward': 0.875, 'std_advantage': 0.27946898341178894} |
| 2026-11-12T06:45:51.446829000Z | grpo_trainer process   0 step      620] loss=0.00043459988 aux={'kl_divergence': 0.01190185546875, 'loss': 0.00043459987500682473, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.28637051582336426} |
| 2026-11-12T06:49:00.160351000Z | grpo_trainer process   0 step      630] loss=0.0004346456 aux={'kl_divergence': 0.01123046875, 'loss': 0.00043464559712447226, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T06:50:34.403352000Z | grpo_trainer process   0 step      640] loss=0.00042068237 aux={'kl_divergence': 0.01025390625, 'loss': 0.00042068236507475376, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.249964639544487} |
| 2026-11-12T06:52:09.491146000Z | grpo_trainer process   0 step      650] loss=0.00053111254 aux={'kl_divergence': 0.01348876953125, 'loss': 0.0005311125423759222, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.24202722311019897} |
| 2026-11-12T08:25:39.190841000Z | grpo_trainer process   0 step      660] loss=0.0019389899 aux={'kl_divergence': 0.01806640625, 'loss': 0.0019389898516237736, 'mean_advantage': 0.0, 'mean_reward': 0.84765625, 'std_advantage': 0.24202722311019897} |
| 2026-11-12T08:27:14.350500000Z | grpo_trainer process   0 step      670] loss=0.0007661622 aux={'kl_divergence': 0.01806640625, 'loss': 0.000766162178479135, 'mean_advantage': 0.0, 'mean_reward': 0.8671875, 'std_advantage': 0.3186436593532562} |
| 2026-11-12T08:30:28.436387000Z | grpo_trainer process   0 step      680] loss=0.0004619468 aux={'kl_divergence': 0.01116943359375, 'loss': 0.0004619467945303768, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.2576576769351959} |
| 2026-11-12T08:32:03.560912000Z | grpo_trainer process   0 step      690] loss=0.0025127137 aux={'kl_divergence': 0.016357421875, 'loss': 0.002512713661417365, 'mean_advantage': 0.0, 'mean_reward': 0.87109375, 'std_advantage': 0.28637048602104187} |
| 2026-11-12T08:33:38.768237000Z | grpo_trainer process   0 step      700] loss=0.00051284337 aux={'kl_divergence': 0.012451171875, 'loss': 0.0005128433695062995, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.2576576769351959} |
| 2026-11-12T10:18:31.539015000Z | grpo_trainer process   0 step      710] loss=0.0005488963 aux={'kl_divergence': 0.0137939453125, 'loss': 0.0005488963215611875, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.3061429262161255} |
| 2026-11-12T10:20:06.968169000Z | grpo_trainer process   0 step      720] loss=0.00041914222 aux={'kl_divergence': 0.01080322265625, 'loss': 0.00041914221947081387, 'mean_advantage': 0.0, 'mean_reward': 0.90625, 'std_advantage': 0.265127569437027} |
| 2026-11-12T10:23:15.840931000Z | grpo_trainer process   0 step      730] loss=0.00051802874 aux={'kl_divergence': 0.01336669921875, 'loss': 0.0005180287407711148, 'mean_advantage': 0.0, 'mean_reward': 0.8828125, 'std_advantage': 0.3061429262161255} |
| 2026-11-12T10:24:51.308469000Z | grpo_trainer process   0 step      740] loss=0.00057203666 aux={'kl_divergence': 0.012939453125, 'loss': 0.0005720366607420146, 'mean_advantage': 0.0, 'mean_reward': 0.88671875, 'std_advantage': 0.2723926603794098} |
| 2026-11-12T10:26:26.907534000Z | grpo_trainer process   0 step      750] loss=0.0026157203 aux={'kl_divergence': 0.0123291015625, 'loss': 0.0026157202664762735, 'mean_advantage': 0.0, 'mean_reward': 0.8828125, 'std_advantage': 0.265127569437027} |
| 2026-11-12T10:29:24.580320000Z | grpo_trainer process   0 step      760] loss=0.00047622627 aux={'kl_divergence': 0.01416015625, 'loss': 0.00047622626880183816, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.2931095361709595} |
| 2026-11-12T10:31:00.034356000Z | grpo_trainer process   0 step      770] loss=0.0005682735 aux={'kl_divergence': 0.01470947265625, 'loss': 0.0005682734772562981, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.23382052779197693} |
| 2026-11-12T10:34:01.258036000Z | grpo_trainer process   0 step      780] loss=0.00055581576 aux={'kl_divergence': 0.01324462890625, 'loss': 0.0005558157572522759, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.2499646693468094} |
| 2026-11-12T10:35:37.289537000Z | grpo_trainer process   0 step      790] loss=0.0025342891 aux={'kl_divergence': 0.01287841796875, 'loss': 0.0025342891458421946, 'mean_advantage': 0.0, 'mean_reward': 0.8828125, 'std_advantage': 0.27946901321411133} |
| 2026-11-12T10:37:12.733139000Z | grpo_trainer process   0 step      800] loss=0.00033631801 aux={'kl_divergence': 0.017578125, 'loss': 0.0003363180148880929, 'mean_advantage': 0.0, 'mean_reward': 0.8671875, 'std_advantage': 0.33067217469215393} |
| 2026-11-12T11:36:40.265290000Z | grpo_trainer process   0 step      810] loss=0.00069567066 aux={'kl_divergence': 0.016845703125, 'loss': 0.000695670663844794, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.2996971011161804} |
| 2026-11-12T11:38:15.963297000Z | grpo_trainer process   0 step      820] loss=0.00069489534 aux={'kl_divergence': 0.0159912109375, 'loss': 0.0006948953378014266, 'mean_advantage': 0.0, 'mean_reward': 0.9140625, 'std_advantage': 0.17675170302391052} |
| 2026-11-12T13:00:25.421711000Z | grpo_trainer process   0 step      830] loss=0.0006183362 aux={'kl_divergence': 0.0155029296875, 'loss': 0.0006183361983858049, 'mean_advantage': 0.0, 'mean_reward': 0.87109375, 'std_advantage': 0.2723926901817322} |
| 2026-11-12T13:02:00.572108000Z | grpo_trainer process   0 step      840] loss=0.0022852055 aux={'kl_divergence': 0.015380859375, 'loss': 0.002285205526277423, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.23382052779197693} |
| 2026-11-12T13:03:35.967792000Z | grpo_trainer process   0 step      850] loss=0.002446006 aux={'kl_divergence': 0.0145263671875, 'loss': 0.002446006052196026, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.265127569437027} |
| 2026-11-12T13:06:46.194789000Z | grpo_trainer process   0 step      860] loss=0.0021097094 aux={'kl_divergence': 0.01611328125, 'loss': 0.0021097094286233187, 'mean_advantage': 0.0, 'mean_reward': 0.8984375, 'std_advantage': 0.2499646544456482} |
| 2026-11-12T13:08:21.585745000Z | grpo_trainer process   0 step      870] loss=0.00025139417 aux={'kl_divergence': 0.01287841796875, 'loss': 0.00025139417266473174, 'mean_advantage': 0.0, 'mean_reward': 0.8671875, 'std_advantage': 0.249964639544487} |
| 2026-11-12T13:11:25.785721000Z | grpo_trainer process   0 step      880] loss=0.002853879 aux={'kl_divergence': 0.0174560546875, 'loss': 0.0028538790065795183, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.2576576769351959} |
| 2026-11-12T13:13:00.238132000Z | grpo_trainer process   0 step      890] loss=0.00076265034 aux={'kl_divergence': 0.01708984375, 'loss': 0.0007626503356732428, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.28637048602104187} |
| 2026-11-12T13:14:35.494811000Z | grpo_trainer process   0 step      900] loss=0.0006159399 aux={'kl_divergence': 0.016357421875, 'loss': 0.0006159399054013193, 'mean_advantage': 0.0, 'mean_reward': 0.8828125, 'std_advantage': 0.23382051289081573} |
| 2026-11-12T14:54:11.188840000Z | grpo_trainer process   0 step      910] loss=0.001025387 aux={'kl_divergence': 0.0130615234375, 'loss': 0.0010253869695588946, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.2931095361709595} |
| 2026-11-12T14:55:46.585015000Z | grpo_trainer process   0 step      920] loss=0.0015594302 aux={'kl_divergence': 0.016357421875, 'loss': 0.001559430151246488, 'mean_advantage': 0.0, 'mean_reward': 0.875, 'std_advantage': 0.265127569437027} |
| 2026-11-12T14:58:51.205224000Z | grpo_trainer process   0 step      930] loss=0.0011953758 aux={'kl_divergence': 0.0159912109375, 'loss': 0.001195375807583332, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.2723926901817322} |
| 2026-11-12T15:00:26.372128000Z | grpo_trainer process   0 step      940] loss=0.0027401117 aux={'kl_divergence': 0.016845703125, 'loss': 0.002740111667662859, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.2576576769351959} |
| 2026-11-12T15:02:01.852052000Z | grpo_trainer process   0 step      950] loss=0.000721977 aux={'kl_divergence': 0.0181884765625, 'loss': 0.0007219769759103656, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.18747349083423615} |
| 2026-11-12T18:40:29.420401000Z | grpo_trainer process   0 step      960] loss=0.0013027997 aux={'kl_divergence': 0.01611328125, 'loss': 0.0013027996756136417, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.3061429262161255} |
| 2026-11-12T18:42:04.395798000Z | grpo_trainer process   0 step      970] loss=0.0015520466 aux={'kl_divergence': 0.0155029296875, 'loss': 0.0015520466258749366, 'mean_advantage': 0.0, 'mean_reward': 0.91015625, 'std_advantage': 0.2723926603794098} |
| 2026-11-12T18:45:06.745087000Z | grpo_trainer process   0 step      980] loss=0.0016491248 aux={'kl_divergence': 0.017578125, 'loss': 0.0016491247806698084, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.2931095361709595} |
| 2026-11-12T18:46:41.680837000Z | grpo_trainer process   0 step      990] loss=0.00058557 aux={'kl_divergence': 0.015869140625, 'loss': 0.0005855700001120567, 'mean_advantage': 0.0, 'mean_reward': 0.8828125, 'std_advantage': 0.2499646544456482} |
| 2026-11-12T18:48:16.528728000Z | grpo_trainer process   0 step     1000] loss=0.0013191977 aux={'kl_divergence': 0.0159912109375, 'loss': 0.0013191977050155401, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.265127569437027} |
