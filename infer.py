import argparse

import tensorflow as tf

from generate_data import SumTaskData


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)

    max_seq_len_placeholder_name = 'prefix/root/Placeholder:0'
    inputs_placeholder_name = 'prefix/root/Placeholder_1:0'
    output_name = 'prefix/root/Sigmoid:0'

    inputs_placeholder = graph.get_tensor_by_name(inputs_placeholder_name)
    max_seq_len_placeholder = graph.get_tensor_by_name(max_seq_len_placeholder_name)

    y = graph.get_tensor_by_name(output_name)

    data_generator = SumTaskData()
    seq_len, inputs, labels = data_generator.generate_batches(
        num_batches=1,
        batch_size=32,
        bits_per_vector=3,
        curriculum_point=None,
        max_seq_len=4,
        curriculum='none',
        pad_to_max_seq_len=False
    )[0]

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
                                           inputs_placeholder: inputs,
                                           # outputs_placeholder: labels,
                                           max_seq_len_placeholder: seq_len
                                       })
        print(y_out.shape)
