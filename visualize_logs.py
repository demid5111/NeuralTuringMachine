import argparse
import re

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def get_img_folder_path():
    project_root = Path(__file__).resolve().parent
    tmp_artifacts_root = project_root / 'artifacts'
    return tmp_artifacts_root


def create_chart(series, ticks, granularity=1000, type='error', bits_per_number=4):
    font_settings = {
        'fontname': 'Times New Roman',
        'fontsize': 12
    }

    if type == 'error':
        y_title = 'error'
        chart_title = 'Error per sequence'
        legend_title = 'Error'
    else:  # it is loss
        y_title = 'loss'
        chart_title = 'Total loss'
        legend_title = 'Loss'

    new_ticks = [i for i in ticks if i % granularity == 0]
    filtered_data_df = pd.DataFrame({
        'x': ticks,
        'data': series,
    })
    filtered_data_df = filtered_data_df[filtered_data_df['x'] <= new_ticks[-1]]

    plt.close()
    plt.plot('x', 'data', data=filtered_data_df, color='black',
             linewidth=2, label=legend_title)
    # plt.yscale('log')
    plt.ylabel(y_title, **font_settings)
    plt.xlabel('training steps', **font_settings)
    plt.xticks(new_ticks, **font_settings)
    plt.yticks(**font_settings)
    plt.title(f'Binary sum task of two {bits_per_number}-bit numbers, {chart_title}')
    plt.legend()
    return plt


def save_chart(plot, type, bits_per_number):
    project_root = Path(__file__).resolve().parent
    tmp_artifacts_root = project_root / 'artifacts'
    try:
        tmp_artifacts_root.mkdir()
    except FileExistsError:
        pass

    path_template = tmp_artifacts_root / f'{bits_per_number}_bit_{type}.eps'
    plot.savefig(path_template, format='eps', dpi=600)


PARSABLE_PATTERN = re.compile(r'''
                        (.*:EVAL_PARSABLE:\s)
                        (?P<step>\d+),
                        (?P<error>\d+\.\d+(e[+-]\d+)?),
                        (?P<loss>\d+\.\d+(e[+-]\d+)?).*''', re.VERBOSE)


def get_history(log_path):
    steps = []
    errors = []
    losses = []
    with open(log_path) as f:
        for line in f:
            matched = re.match(PARSABLE_PATTERN, line)
            if not matched:
                continue
            steps.append(int(matched.group('step')))
            errors.append(float(matched.group('error')))
            losses.append(float(matched.group('loss')))
    return steps, errors, losses


def main(args):
    steps, errors, losses = get_history(args.log_path)
    plot = create_chart(series=errors, ticks=steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)
    plot = create_chart(series=losses, ticks=steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)
    # visualize_error(errors)
    # losses = get_loss_history()
    # visualize_loss(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', required=True, type=str,
                        help='Path to training log')
    parser.add_argument('--granularity', required=False, type=int, default=5000,
                        help='Granularity for ticks')
    parser.add_argument('--bits_per_number', required=True, type=int,
                        help='Bits per number')
    main(parser.parse_args())
