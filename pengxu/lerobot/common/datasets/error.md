(openvla-oft) pengxu@bld-SP2C621D:/data1/pengxu/minivla-oft-main/openvla-oft$ sh train_realworld.sh 
RMSNorm 补丁已应用
当前时间是: 2025-07-29_19-43-35
/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
2025-07-29 19:43:39.763871: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-29 19:43:39.804302: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-07-29 19:43:39.804338: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-07-29 19:43:39.805754: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-07-29 19:43:39.812310: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-07-29 19:43:40.529285: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Using REALWORLD constants:
  NUM_ACTIONS_CHUNK = 8
  ACTION_DIM = 7
  PROPRIO_DIM = 7
  ACTION_PROPRIO_NORMALIZATION_TYPE = bounds_q99
If needed, manually set the correct constants in `prismatic/vla/constants.py`!
Fine-tuning OpenVLA Model `/data1/pengxu/minivla-oft-main/pretrained_models/minivla` on `grab_cube`
wandb: Tracking run with wandb version 0.21.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
Detected constants:
        NUM_ACTIONS_CHUNK: 8
        ACTION_DIM: 7
        PROPRIO_DIM: 7
        ACTION_PROPRIO_NORMALIZATION_TYPE: bounds_q99
Created backup of original config at: /data1/pengxu/minivla-oft-main/pretrained_models/minivla/config.json.back.20250729_194352
Updated config.json at: /data1/pengxu/minivla-oft-main/pretrained_models/minivla/config.json
Changes made:
  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"
  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"
07/29 [19:43:53] INFO     | >> [*] Loading from local path                                            load.py:62
                          `/data1/pengxu/minivla-oft-main/pretrained_models/prism-qwen25-extra-dinosi           
                          glip-224px-0_5b/pretrained_models`                                                    
                 INFO     | >> [*] Found Config =>> Loading & Freezing                                load.py:85
                          prism-qwen25-extra-dinosiglip-224px+0_5b with:                                        
                                       Vision Backbone =>> dinosiglip-vit-so-224px                              
                                       LLM Backbone    =>> qwen25-0_5b-extra                                    
                                       Arch Specifier  =>> no-align+fused-gelu-mlp                              
                                       Checkpoint Path =>>                                                      
                          `/data1/pengxu/minivla-oft-main/pretrained_models/prism-qwen25-extra-dinosi           
                          glip-224px-0_5b/pretrained_models/checkpoints/step-020792-epoch-01-loss=0.5           
                          268.pt`                                                                               
                 INFO     | >> [*] Loading Vision Backbone dinosiglip-vit-so-224px                   load.py:100
07/29 [19:43:57] INFO     | >> Loading pretrained weights from Hugging Face hub                  _builder.py:186
                          (timm/vit_large_patch14_reg4_dinov2.lvd142m)                                          
                 INFO     | >>  Safe alternative available for 'pytorch_model.bin' (as               _hub.py:180
                          'model.safetensors'). Loading weights using safetensors.                              
                 INFO     | >> Resized position embedding: (37, 37) to (16, 16).                 pos_embed.py:55
07/29 [19:44:02] INFO     | >> Loading pretrained weights from Hugging Face hub                  _builder.py:186
                          (('timm/ViT-SO400M-14-SigLIP', 'open_clip_pytorch_model.bin'))                        
                 INFO     | >>  Safe alternative available for 'open_clip_pytorch_model.bin' (as     _hub.py:180
                          'open_clip_model.safetensors'). Loading weights using safetensors.                    
                 INFO     | >> [*] Loading Pretrained LLM qwen25-0_5b-extra via HF Transformers      load.py:108
                 INFO     | >>     |=> Loading qwen2.5 LLM from `Qwen/Qwen2.5-0.5B`              base_llm.py:121
                 INFO     | >>     |=> Loading qwen2.5 (Fast) Tokenizer via the AutoTokenizer    base_llm.py:156
                          API                                                                                   
Added 256 extra tokens.
                 INFO     | >> [*] Loading VLM prism-qwen25-extra-dinosiglip-224px+0_5b from         load.py:117
                          Checkpoint                                                                            
trainable params: 103,851,520 || all params: 1,356,347,968 || trainable%: 7.6567
# trainable params in proprio_projector: 810880
# trainable params in action_head: 102129695
# total trainable params: 206792095
2025-07-29 19:45:07.184517: I tensorflow/core/grappler/optimizers/data/replicate_on_split.cc:32] Running replicate on split optimization
07/29 [19:45:07] INFO     | >> [*] Computing dataset statistics. This may take a bit, but      data_utils.py:223
                          should only need to happen once.                                                      
100%|████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 426.10it/s]
2025-07-29 19:45:08.260141: I tensorflow/core/grappler/optimizers/data/replicate_on_split.cc:32] Running replicate on split optimization
Traceback (most recent call last):
  File "/data1/pengxu/minivla-oft-main/openvla-oft/vla-scripts/finetune.py", line 1613, in <module>
    finetune()
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/draccus/argparsing.py", line 203, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/data1/pengxu/minivla-oft-main/openvla-oft/vla-scripts/finetune.py", line 1445, in finetune
    train_dataset = RLDSDataset(
  File "/data1/pengxu/minivla-oft-main/openvla-oft/prismatic/vla/datasets/datasets.py", line 203, in __init__
    self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
  File "/data1/pengxu/minivla-oft-main/openvla-oft/prismatic/vla/datasets/datasets.py", line 206, in make_dataset
    return make_interleaved_dataset(**rlds_config)
  File "/data1/pengxu/minivla-oft-main/openvla-oft/prismatic/vla/datasets/rlds/dataset.py", line 507, in make_interleaved_dataset
    _, dataset_statistics = make_dataset_from_rlds(**data_kwargs, train=train)
  File "/data1/pengxu/minivla-oft-main/openvla-oft/prismatic/vla/datasets/rlds/dataset.py", line 239, in make_dataset_from_rlds
    dataset = dataset.traj_map(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/dlimp/dataset.py", line 17, in wrapper
    result = f(*args, **kwargs)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/dlimp/dataset.py", line 178, in traj_map
    return super().map(fn, num_parallel_calls=num_parallel_calls, **kwargs)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/data/ops/dataset_ops.py", line 2280, in map
    return map_op._map_v2(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/data/ops/map_op.py", line 40, in _map_v2
    return _ParallelMapDataset(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/data/ops/map_op.py", line 148, in __init__
    self._map_func = structured_function.StructuredFunctionWrapper(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/data/ops/structured_function.py", line 265, in __init__
    self._function = fn_factory()
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 1227, in get_concrete_function
    concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 1197, in _get_concrete_function_garbage_collected
    self._initialize(args, kwargs, add_initializers_to=initializers)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 695, in _initialize
    self._concrete_variable_creation_fn = tracing_compilation.trace_function(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py", line 178, in trace_function
    concrete_function = _maybe_define_function(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py", line 283, in _maybe_define_function
    concrete_function = _create_concrete_function(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py", line 310, in _create_concrete_function
    traced_func_graph = func_graph_module.func_graph_from_py_func(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/framework/func_graph.py", line 1059, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 598, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/data/ops/structured_function.py", line 231, in wrapped_fn
    ret = wrapper_helper(*args)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/data/ops/structured_function.py", line 161, in wrapper_helper
    ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 693, in wrapper
    raise e.ag_error_metadata.to_exception(e)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 690, in wrapper
    return converted_call(f, args, kwargs, options=options)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 352, in converted_call
    return converted_call(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 439, in converted_call
    result = converted_f(*effective_args, **kwargs)
  File "/tmp/__autograph_generated_filevaf2qg4d.py", line 149, in tf__normalize_action_and_proprio
    ag__.if_stmt(ag__.ld(normalization_type) == ag__.ld(NormalizationType).NORMAL, if_body_4, else_body_4, get_state_6, set_state_6, ('do_return', 'retval_', 'high', 'low', 'traj'), 2)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1217, in if_stmt
    _py_if_stmt(cond, body, orelse)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1270, in _py_if_stmt
    return body() if cond else orelse()
  File "/tmp/__autograph_generated_filevaf2qg4d.py", line 129, in else_body_4
    ag__.if_stmt(ag__.ld(normalization_type) in [ag__.ld(NormalizationType).BOUNDS, ag__.ld(NormalizationType).BOUNDS_Q99], if_body_2, else_body_2, get_state_4, set_state_4, ('do_return', 'retval_', 'high', 'low', 'traj'), 2)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1217, in if_stmt
    _py_if_stmt(cond, body, orelse)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1270, in _py_if_stmt
    return body() if cond else orelse()
  File "/tmp/__autograph_generated_filevaf2qg4d.py", line 112, in if_body_2
    ag__.for_stmt(ag__.converted_call(ag__.ld(keys_to_normalize).items, (), None, fscope), None, loop_body_1, get_state_3, set_state_3, ('traj', 'high', 'low'), {'iterate_names': '(key, traj_key)'})
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 449, in for_stmt
    for_fn(iter_, extra_test, body, get_state, set_state, symbol_names, opts)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 500, in _py_for_stmt
    body(target)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 466, in protected_body
    original_body(protected_iter)
  File "/tmp/__autograph_generated_filevaf2qg4d.py", line 103, in loop_body_1
    traj = ag__.converted_call(ag__.ld(dl).transforms.selective_tree_map, (ag__.ld(traj),), dict(match=ag__.autograph_artifact(lambda k, _: ag__.ld(k) == ag__.ld(traj_key)), map_fn=ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.ld(tf).where, (ag__.ld(mask), ag__.converted_call(ag__.ld(tf).clip_by_value, (2 * (ag__.ld(x) - ag__.ld(low)) / (ag__.ld(high) - ag__.ld(low) + 1e-08) - 1, -1, 1), None, fscope), ag__.ld(x)), None, fscope))), fscope)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 439, in converted_call
    result = converted_f(*effective_args, **kwargs)
  File "/tmp/__autograph_generated_file0q_dy79k.py", line 80, in tf__selective_tree_map
    ag__.for_stmt(ag__.ld(x), None, loop_body, get_state_3, set_state_3, (), {'iterate_names': 'key'})
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 449, in for_stmt
    for_fn(iter_, extra_test, body, get_state, set_state, symbol_names, opts)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 500, in _py_for_stmt
    body(target)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 466, in protected_body
    original_body(protected_iter)
  File "/tmp/__autograph_generated_file0q_dy79k.py", line 78, in loop_body
    ag__.if_stmt(ag__.converted_call(ag__.ld(isinstance), (ag__.ld(x)[ag__.ld(key)], ag__.ld(dict)), None, fscope), if_body_2, else_body_2, get_state_2, set_state_2, ('out[key]',), 1)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1217, in if_stmt
    _py_if_stmt(cond, body, orelse)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1270, in _py_if_stmt
    return body() if cond else orelse()
  File "/tmp/__autograph_generated_file0q_dy79k.py", line 77, in else_body_2
    ag__.if_stmt(ag__.converted_call(ag__.ld(match_fn), (ag__.ld(_keypath) + ag__.ld(key), ag__.ld(x)[ag__.ld(key)]), None, fscope), if_body_1, else_body_1, get_state_1, set_state_1, ('out[key]',), 1)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1217, in if_stmt
    _py_if_stmt(cond, body, orelse)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/operators/control_flow.py", line 1270, in _py_if_stmt
    return body() if cond else orelse()
  File "/tmp/__autograph_generated_file0q_dy79k.py", line 73, in if_body_1
    ag__.ld(out)[ag__.ld(key)] = ag__.converted_call(ag__.ld(map_fn), (ag__.ld(x)[ag__.ld(key)],), None, fscope)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 339, in converted_call
    return _call_unconverted(f, args, kwargs, options)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 460, in _call_unconverted
    return f(*args)
  File "/tmp/__autograph_generated_filevaf2qg4d.py", line 103, in <lambda>
    traj = ag__.converted_call(ag__.ld(dl).transforms.selective_tree_map, (ag__.ld(traj),), dict(match=ag__.autograph_artifact(lambda k, _: ag__.ld(k) == ag__.ld(traj_key)), map_fn=ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.ld(tf).where, (ag__.ld(mask), ag__.converted_call(ag__.ld(tf).clip_by_value, (2 * (ag__.ld(x) - ag__.ld(low)) / (ag__.ld(high) - ag__.ld(low) + 1e-08) - 1, -1, 1), None, fscope), ag__.ld(x)), None, fscope))), fscope)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 377, in converted_call
    return _call_unconverted(f, args, kwargs, options)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/autograph/impl/api.py", line 460, in _call_unconverted
    return f(*args)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/tensorflow/python/framework/ops.py", line 1020, in _create_c_op
    raise ValueError(e.message)
ValueError: in user code:

    File "/data1/pengxu/minivla-oft-main/openvla-oft/prismatic/vla/datasets/rlds/utils/data_utils.py", line 76, in normalize_action_and_proprio  *
        traj = dl.transforms.selective_tree_map(
    File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/dlimp/transforms/common.py", line 41, in selective_tree_map  *
        out[key] = map_fn(x[key])
    File "/tmp/__autograph_generated_filevaf2qg4d.py", line 103, in <lambda>  **
        traj = ag__.converted_call(ag__.ld(dl).transforms.selective_tree_map, (ag__.ld(traj),), dict(match=ag__.autograph_artifact(lambda k, _: ag__.ld(k) == ag__.ld(traj_key)), map_fn=ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.ld(tf).where, (ag__.ld(mask), ag__.converted_call(ag__.ld(tf).clip_by_value, (2 * (ag__.ld(x) - ag__.ld(low)) / (ag__.ld(high) - ag__.ld(low) + 1e-08) - 1, -1, 1), None, fscope), ag__.ld(x)), None, fscope))), fscope)

    ValueError: Dimensions must be equal, but are 7 and 8 for '{{node SelectV2}} = SelectV2[T=DT_FLOAT](SelectV2/condition, clip_by_value, args_1)' with input shapes: [7], [?,8], [?,8].

wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /data1/pengxu/minivla-oft-main/openvla-oft/wandb/offline-run-20250729_194352-d4jdvak1
wandb: Find logs at: wandb/offline-run-20250729_194352-d4jdvak1/logs
[2025-07-29 19:45:12,136] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 380266) of binary: /data0/pengxu/.conda/envs/openvla-oft/bin/python3.10
Traceback (most recent call last):
  File "/data0/pengxu/.conda/envs/openvla-oft/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/torch/distributed/run.py", line 812, in main
    run(args)
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/torch/distributed/run.py", line 803, in run
    elastic_launch(
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 135, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/data0/pengxu/.conda/envs/openvla-oft/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 268, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
vla-scripts/finetune.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-07-29_19:45:12
  host      : bld-SP2C621D
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 380266)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================