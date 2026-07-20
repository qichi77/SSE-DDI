import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS = [
    'acc',
    'auc',
    'f1',
    'precision',
    'recall',
    'ap',
]


def run_command(command):
    print('\n$', ' '.join(command))
    subprocess.run(
        command,
        check=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        choices=['drugbank', 'twosides'],
        required=True,
    )
    parser.add_argument(
        '--mode',
        choices=['transductive', 'inductive'],
        required=True,
    )
    parser.add_argument(
        '--raw-file',
        required=True,
    )
    parser.add_argument(
        '--output-root',
        default='./data/processed',
    )
    parser.add_argument(
        '--result-root',
        default='./results',
    )
    parser.add_argument(
        '--checkpoint-root',
        default='./checkpoints',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        '--skip-drug-data',
        action='store_true',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
    )

    return parser.parse_args()


def aggregate_results(
    result_files,
    mode,
):
    if mode == 'transductive':
        scenarios = ['test']
    else:
        scenarios = ['s1', 's2']

    all_results = []

    for result_file in result_files:
        with result_file.open(
            'r',
            encoding='utf-8',
        ) as file:
            all_results.append(json.load(file))

    summary = {}

    for scenario in scenarios:
        summary[scenario] = {}

        for metric in METRICS:
            values = np.asarray(
                [
                    result[scenario][metric]
                    for result in all_results
                ],
                dtype=np.float64,
            )

            summary[scenario][metric] = {
                'mean': float(
                    np.nanmean(values)
                ),
                # 与论文 mean ± std 对应，采用样本标准差。
                'std': float(
                    np.nanstd(values, ddof=1)
                ),
                'values': values.tolist(),
            }

    return summary


def main():
    args = parse_args()

    python = sys.executable

    if not args.skip_drug_data:
        run_command([
            python,
            'data_pre.py',
            '--dataset',
            args.dataset,
            '--raw-file',
            args.raw_file,
            '--output-root',
            args.output_root,
            '--operation',
            'drug_data',
        ])

    result_files = []

    for seed in args.seeds:
        run_command([
            python,
            'data_pre.py',
            '--dataset',
            args.dataset,
            '--raw-file',
            args.raw_file,
            '--output-root',
            args.output_root,
            '--operation',
            'split',
            '--mode',
            args.mode,
            '--seed',
            str(seed),
        ])

        run_command([
            python,
            'train.py',
            '--dataset',
            args.dataset,
            '--mode',
            args.mode,
            '--seed',
            str(seed),
            '--data-root',
            args.output_root,
            '--result-root',
            args.result_root,
            '--checkpoint-root',
            args.checkpoint_root,
            '--num-workers',
            str(args.num_workers),
        ])

        result_files.append(
            Path(args.result_root)
            / args.dataset
            / args.mode
            / f'seed{seed}.json'
        )

    summary = aggregate_results(
        result_files=result_files,
        mode=args.mode,
    )

    output = {
        'dataset': args.dataset,
        'mode': args.mode,
        'seeds': args.seeds,
        'summary': summary,
    }

    summary_file = (
        Path(args.result_root)
        / args.dataset
        / args.mode
        / 'summary.json'
    )
    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_file.open(
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print('\nFinal summary')

    for scenario, scenario_values in summary.items():
        print(f'\n[{scenario.upper()}]')

        for metric, values in scenario_values.items():
            print(
                f'{metric:10s}: '
                f'{values["mean"]:.4f} '
                f'± {values["std"]:.4f}'
            )

    print(f'\nSummary saved to {summary_file}')


if __name__ == '__main__':
    main()
