# Training information

## General

Date: TBD
Task: sum (binary numbers)

## CLI command

```bash
TBD
```

## Logs

```bash
Using local implementation
WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

INFO:numexpr.utils:NumExpr defaulting to 2 threads.
WARNING:tensorflow:From run_tasks.py:202: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:202: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:203: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From run_tasks.py:203: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/ntm.py:31: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/ntm.py:31: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/ntm.py:33: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/ntm.py:33: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/utils.py:29: calling TruncatedNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/utils.py:29: calling TruncatedNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From run_tasks.py:144: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From run_tasks.py:144: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/contrib/layers/python/layers/layers.py:1866: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/contrib/layers/python/layers/layers.py:1866: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/rnn_cell_impl.py:735: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/rnn_cell_impl.py:735: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/rnn_cell_impl.py:739: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/rnn_cell_impl.py:739: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/ntm.py:117: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/ntm.py:117: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/util/dispatch.py:180: calling expand_dims (from tensorflow.python.ops.array_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/util/dispatch.py:180: calling expand_dims (from tensorflow.python.ops.array_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/ops/nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From run_tasks.py:178: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:178: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:180: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:180: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:207: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:207: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:209: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:209: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:257: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From run_tasks.py:257: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2021-02-19 13:23:40.311992: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-02-19 13:23:40.323541: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-02-19 13:23:40.323608: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (960c3885f5a2): /proc/driver/nvidia/version does not exist
2021-02-19 13:23:40.343149: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299995000 Hz
2021-02-19 13:23:40.343577: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x161cd80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-02-19 13:23:40.343624: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 0.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

Tensorflow reading models/0/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/0/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/0/my_model.ckpt
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:40: convert_variables_to_constants (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.convert_variables_to_constants`
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:40: convert_variables_to_constants (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.convert_variables_to_constants`
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/framework/graph_util_impl.py:277: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/framework/graph_util_impl.py:277: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 0.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

INFO:utils:Tested frozen model at step 0, error: 5.84375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.615625,22.534457397460937
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 0,5.615625,22.534457397460937,None,None,None,None,None
Current curriculum point: None
None
1.0
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
WARNING:tensorflow:Ignoring: models/0; No such file or directory
WARNING:tensorflow:Ignoring: models/0; No such file or directory
INFO:utils:Saved the trained model at step 1000.
Tensorflow reading models/1000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/1000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/1000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 1000.
INFO:utils:Tested frozen model at step 1000, error: 5.375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.528125,7.629039239883423
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 1000,5.528125,7.629039239883423,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 2000.
Tensorflow reading models/2000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/2000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/2000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 2000.
INFO:utils:Tested frozen model at step 2000, error: 5.625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.35625,7.621520376205444
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 2000,5.35625,7.621520376205444,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 3000.
Tensorflow reading models/3000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/3000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/3000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 3000.
INFO:utils:Tested frozen model at step 3000, error: 5.15625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.603125,7.627658557891846
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 3000,5.603125,7.627658557891846,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 4000.
Tensorflow reading models/4000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/4000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/4000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 4000.
INFO:utils:Tested frozen model at step 4000, error: 5.5.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.4875,7.6256335258483885
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 4000,5.4875,7.6256335258483885,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 5000.
Tensorflow reading models/5000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/5000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/5000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 5000.
INFO:utils:Tested frozen model at step 5000, error: 5.65625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.48125,7.626587867736816
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 5000,5.48125,7.626587867736816,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 6000.
Tensorflow reading models/6000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/6000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/6000/my_model.ckpt
^C
time: 1h 20min 49s (started: 2021-02-19 13:23:33 +00:00)

```