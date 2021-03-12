import argparse
from pathlib import Path

import tensorflow as tf

from tasks.arithmetics.binary_average_sum.generator import AverageSumTaskData
from tasks.arithmetics.common.binary_arithmetics import BinaryUtils
from tasks.arithmetics.binary_sum.generator import SumTaskData


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='prefix')
    return graph


def prepare_graph_for_inference(directory_path: Path):
    graph = load_graph(str(directory_path / 'frozen_graph.pb'))

    max_seq_len_placeholder_name = 'prefix/root/Placeholder:0'
    inputs_placeholder_name = 'prefix/root/Placeholder_1:0'
    output_name = 'prefix/root/Sigmoid:0'

    inputs_placeholder = graph.get_tensor_by_name(inputs_placeholder_name)
    max_seq_len_placeholder = graph.get_tensor_by_name(max_seq_len_placeholder_name)

    y = graph.get_tensor_by_name(output_name)

    return graph, (
        inputs_placeholder,
        max_seq_len_placeholder
    ), y


def infer_model(directory_path: Path, inputs, seq_len):
    graph, (inputs_placeholder, seq_len_placeholder), y = prepare_graph_for_inference(directory_path)
    with tf.Session(graph=graph) as sess:
        outputs = sess.run(y, feed_dict={
            inputs_placeholder: inputs,
            seq_len_placeholder: seq_len
        })
    return outputs


def _generate_data(bits_per_number, num_experts):
    generator_args = dict(
        num_batches=1,
        batch_size=32,
        bits_per_vector=3,
        curriculum_point=None,
        max_seq_len=bits_per_number,
        curriculum='none',
        pad_to_max_seq_len=False
    )

    if num_experts is None:
        data_generator = SumTaskData()
    else:
        data_generator = AverageSumTaskData()
        generator_args['numbers_quantity'] = args.num_experts

    return data_generator.generate_batches(**generator_args)[0], data_generator


def test_model(directory_path: Path, bits_per_number, num_experts):
    (seq_len, inputs, labels), data_generator = _generate_data(bits_per_number, num_experts)

    outputs = infer_model(directory_path, inputs=inputs, seq_len=seq_len)

    error = data_generator.error_per_seq(labels, outputs, 32)

    return error


def demo_summator(directory_path: Path, a: int, b: int, bits_per_number, num_experts):
    (seq_len, inputs, labels), data_generator = _generate_data(bits_per_number, num_experts)

    a_numpy = BinaryUtils._from_decimal_to_little_endian_binary_numpy(a, bits_per_number=bits_per_number)
    b_numpy = BinaryUtils._from_decimal_to_little_endian_binary_numpy(b, bits_per_number=bits_per_number)

    inputs[0][:bits_per_number, 0] = a_numpy
    inputs[0][bits_per_number + 1:bits_per_number * 2 + 1, 0] = b_numpy

    outputs = infer_model(directory_path, inputs=inputs, seq_len=seq_len)
    return BinaryUtils._from_binary_numpy_to_decimal(outputs[0][:-1, 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_model_filename', default='results/frozen_model.pb', type=str,
                        help='Frozen model file to import')
    parser.add_argument('--num_experts', required=False, type=int,
                        help='Optional. Needed for average sum task and stands for the quantity of numbers to be used'
                             'for calculations')
    parser.add_argument('--bits_per_number', required=True, default=10, type=int,
                        help='Defines how many bits is allocated for the number representation')

    args = parser.parse_args()

    model = Path(args.frozen_model_filename)

    overall_err = test_model(model.parent, bits_per_number=args.bits_per_number, num_experts=args.num_experts)
    print(f'Overall quality of model. Error: {overall_err}')

    a = 300
    b = 400
    demo_result = demo_summator(model.parent, a, b, bits_per_number=args.bits_per_number, num_experts=args.num_experts)
    print(f'{a} + {b} = {demo_result}')
