from pathlib import Path

import tensorflow as tf


meta_path = Path('./trained_models/binary_sum_v1/my_model.ckpt.meta') # Your .meta file
output_node_names = ['root/Sigmoid']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(str(meta_path))

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('./models'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open(meta_path.parent / 'frozen_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())