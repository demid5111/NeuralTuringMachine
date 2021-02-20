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
WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

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
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
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
WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2021-02-19 16:46:51.660264: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-02-19 16:46:51.671215: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-02-19 16:46:51.671272: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (960c3885f5a2): /proc/driver/nvidia/version does not exist
2021-02-19 16:46:51.677578: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299995000 Hz
2021-02-19 16:46:51.677905: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x29e2d80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-02-19 16:46:51.677939: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Tensorflow reading models/6000/my_model.ckpt checkpoint
INFO:tensorflow:Restoring parameters from models/6000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/6000/my_model.ckpt
Tensorflow loaded models/6000/my_model.ckpt checkpoint
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 7000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

Tensorflow reading models/7000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/7000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/7000/my_model.ckpt
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
INFO:utils:Froze the model at step 7000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

INFO:utils:Tested frozen model at step 7000, error: 5.0.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.40625,7.659900808334351
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 7000,5.40625,7.659900808334351,None,None,None,None,None
Current curriculum point: None
None
1.0
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
INFO:utils:Saved the trained model at step 8000.
Tensorflow reading models/8000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/8000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/8000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 8000.
INFO:utils:Tested frozen model at step 8000, error: 5.0625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.546875,7.657679176330566
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 8000,5.546875,7.657679176330566,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 9000.
Tensorflow reading models/9000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/9000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/9000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 9000.
INFO:utils:Tested frozen model at step 9000, error: 5.53125.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.453125,7.630939865112305
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 9000,5.453125,7.630939865112305,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 10000.
Tensorflow reading models/10000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/10000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/10000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 10000.
INFO:utils:Tested frozen model at step 10000, error: 5.40625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.525,7.628073215484619
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 10000,5.525,7.628073215484619,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 11000.
Tensorflow reading models/11000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/11000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/11000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 11000.
INFO:utils:Tested frozen model at step 11000, error: 5.5.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.375,7.622836780548096
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 11000,5.375,7.622836780548096,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 12000.
Tensorflow reading models/12000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/12000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/12000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 12000.
INFO:utils:Tested frozen model at step 12000, error: 5.75.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.678125,7.62549614906311
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 12000,5.678125,7.62549614906311,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 13000.
Tensorflow reading models/13000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/13000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/13000/my_model.ckpt
^C
time: 1h 37min 31s (started: 2021-02-19 16:46:44 +00:00)
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
WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

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
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
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
WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2021-02-20 05:36:00.540698: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-02-20 05:36:00.613542: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-02-20 05:36:00.613636: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (7cfa4285cd83): /proc/driver/nvidia/version does not exist
2021-02-20 05:36:00.645928: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299995000 Hz
2021-02-20 05:36:00.646323: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1b16d80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-02-20 05:36:00.646367: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Tensorflow reading models/13000/my_model.ckpt checkpoint
INFO:tensorflow:Restoring parameters from models/13000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/13000/my_model.ckpt
Tensorflow loaded models/13000/my_model.ckpt checkpoint
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 14000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

Tensorflow reading models/14000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/14000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/14000/my_model.ckpt
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
INFO:utils:Froze the model at step 14000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

INFO:utils:Tested frozen model at step 14000, error: 5.21875.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.44375,7.6240825176239015
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 14000,5.44375,7.6240825176239015,None,None,None,None,None
Current curriculum point: None
None
1.0
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
INFO:utils:Saved the trained model at step 15000.
Tensorflow reading models/15000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/15000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/15000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 15000.
INFO:utils:Tested frozen model at step 15000, error: 5.25.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.390625,7.623030138015747
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 15000,5.390625,7.623030138015747,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 16000.
Tensorflow reading models/16000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/16000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/16000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 16000.
INFO:utils:Tested frozen model at step 16000, error: 5.5.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.5,7.635226726531982
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 16000,5.5,7.635226726531982,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 17000.
Tensorflow reading models/17000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/17000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/17000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 17000.
INFO:utils:Tested frozen model at step 17000, error: 5.9375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.340625,7.620635843276977
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 17000,5.340625,7.620635843276977,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 18000.
Tensorflow reading models/18000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/18000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/18000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 18000.
INFO:utils:Tested frozen model at step 18000, error: 5.4375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.39375,7.624164676666259
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 18000,5.39375,7.624164676666259,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 19000.
Tensorflow reading models/19000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/19000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/19000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 19000.
INFO:utils:Tested frozen model at step 19000, error: 5.375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.575,7.634746837615967
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 19000,5.575,7.634746837615967,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 20000.
Tensorflow reading models/20000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/20000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/20000/my_model.ckpt
^C
time: 1h 33min 3s (started: 2021-02-20 05:35:53 +00:00)
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
WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

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
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
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
WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2021-02-20 07:22:02.467404: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-02-20 07:22:02.540934: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-02-20 07:22:02.541037: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (7cfa4285cd83): /proc/driver/nvidia/version does not exist
2021-02-20 07:22:02.578353: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299995000 Hz
2021-02-20 07:22:02.578816: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1820d80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-02-20 07:22:02.578863: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Tensorflow reading models/20000/my_model.ckpt checkpoint
INFO:tensorflow:Restoring parameters from models/20000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/20000/my_model.ckpt
Tensorflow loaded models/20000/my_model.ckpt checkpoint
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 21000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

Tensorflow reading models/21000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/21000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/21000/my_model.ckpt
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
INFO:utils:Froze the model at step 21000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

INFO:utils:Tested frozen model at step 21000, error: 5.5.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.44375,7.62417459487915
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 21000,5.44375,7.62417459487915,None,None,None,None,None
Current curriculum point: None
None
1.0
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
INFO:utils:Saved the trained model at step 22000.
Tensorflow reading models/22000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/22000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/22000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 22000.
INFO:utils:Tested frozen model at step 22000, error: 6.125.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.5,7.624710273742676
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 22000,5.5,7.624710273742676,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 23000.
Tensorflow reading models/23000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/23000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/23000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 23000.
INFO:utils:Tested frozen model at step 23000, error: 5.4375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.415625,7.623848390579224
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 23000,5.415625,7.623848390579224,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 24000.
Tensorflow reading models/24000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/24000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/24000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 24000.
INFO:utils:Tested frozen model at step 24000, error: 5.53125.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.478125,7.638778638839722
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 24000,5.478125,7.638778638839722,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 25000.
Tensorflow reading models/25000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/25000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/25000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 25000.
INFO:utils:Tested frozen model at step 25000, error: 5.40625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.54375,7.655452919006348
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 25000,5.54375,7.655452919006348,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 26000.
Tensorflow reading models/26000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/26000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/26000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 26000.
INFO:utils:Tested frozen model at step 26000, error: 5.15625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.428125,7.623463773727417
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 26000,5.428125,7.623463773727417,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 27000.
Tensorflow reading models/27000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/27000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/27000/my_model.ckpt
^C
time: 1h 39min 29s (started: 2021-02-20 07:21:46 +00:00)
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
WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:207: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From run_tasks.py:208: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

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
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From run_tasks.py:149: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
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
WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:183: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:185: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:212: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:260: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From run_tasks.py:261: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2021-02-20 10:36:11.963455: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-02-20 10:36:12.049221: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-02-20 10:36:12.049315: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (7cfa4285cd83): /proc/driver/nvidia/version does not exist
2021-02-20 10:36:12.097045: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299995000 Hz
2021-02-20 10:36:12.097553: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x262cd80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-02-20 10:36:12.097613: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Tensorflow reading models/27000/my_model.ckpt checkpoint
INFO:tensorflow:Restoring parameters from models/27000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/27000/my_model.ckpt
Tensorflow loaded models/27000/my_model.ckpt checkpoint
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 28000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/freeze.py:29: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graph instead.

Tensorflow reading models/28000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/28000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/28000/my_model.ckpt
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
INFO:utils:Froze the model at step 28000.
WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:12: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /content/drive/My Drive/HSE/Учёба/PhD/repositories_for_colab/tf1-approved-NeuralTuringMachine/infer.py:13: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

INFO:utils:Tested frozen model at step 28000, error: 5.59375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.51875,7.632861852645874
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 28000,5.51875,7.632861852645874,None,None,None,None,None
Current curriculum point: None
None
1.0
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
WARNING:tensorflow:From /tensorflow-1.15.2/python3.6/tensorflow_core/python/training/saver.py:963: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
INFO:utils:Saved the trained model at step 29000.
Tensorflow reading models/29000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/29000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/29000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 29000.
INFO:utils:Tested frozen model at step 29000, error: 5.625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.590625,7.629060077667236
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 29000,5.590625,7.629060077667236,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 30000.
Tensorflow reading models/30000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/30000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/30000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 30000.
INFO:utils:Tested frozen model at step 30000, error: 5.625.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.5,7.626175260543823
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 30000,5.5,7.626175260543823,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 31000.
Tensorflow reading models/31000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/31000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/31000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 31000.
INFO:utils:Tested frozen model at step 31000, error: 5.53125.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.415625,7.623981046676636
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 31000,5.415625,7.623981046676636,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 32000.
Tensorflow reading models/32000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/32000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/32000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 32000.
INFO:utils:Tested frozen model at step 32000, error: 5.6875.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.584375,7.626697969436646
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 32000,5.584375,7.626697969436646,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 33000.
Tensorflow reading models/33000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/33000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/33000/my_model.ckpt
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Froze 10 variables.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:tensorflow:Converted 10 variables to const ops.
INFO:utils:Froze the model at step 33000.
INFO:utils:Tested frozen model at step 33000, error: 5.4375.
INFO:utils:----EVAL----
INFO:utils:target task error/loss: 5.559375,7.625366258621216
INFO:utils:multi task error/loss: None,None
INFO:utils:curriculum point error/loss (None): None,None
INFO:utils:EVAL_PARSABLE: 33000,5.559375,7.625366258621216,None,None,None,None,None
Current curriculum point: None
None
1.0
INFO:utils:Saved the trained model at step 34000.
Tensorflow reading models/34000/my_model.ckpt before freezing
INFO:tensorflow:Restoring parameters from models/34000/my_model.ckpt
INFO:tensorflow:Restoring parameters from models/34000/my_model.ckpt
^C
time: 1h 41min 8s (started: 2021-02-20 10:35:53 +00:00)
```