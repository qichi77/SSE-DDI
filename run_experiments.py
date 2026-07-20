import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRIC_NAMES = (
    'acc',
    'auc',
    'f1',
    'precision',
    'recall',
    'ap',
)


def run_command(command):
    print('\n$', ' '.join(command))

    subprocess.run(
        command,
        check=True,
    )


def append_optional_argument(
    command,
    flag,
    value,
):
    if value is not None:
        command.extend(
            [flag, str(value)]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run five independent SSE-DDI experiments '
            'and aggregate mean and standard deviation.'
        )
    )

    parser.add_argument(
        '--dataset',
        choices=['drugbank', 'twosides'],
        required=True,
    )
    parser.add_argument(
        '--mode',
        choices=[
            'transductive',
            'inductive',
        ],
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
        '--unseen-ratio',
        type=float,
        default=0.20,
    )

    parser.add_argument(
        '--sim-topk',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--sim-quantile',
        type=float,
        default=0.70,
    )
    parser.add_argument(
        '--sim-std-lambda',
        type=float,
        default=0.50,
    )
    parser.add_argument(
        '--d-min',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--d-max',
        type=int,
        default=64,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--lr-gamma',
        type=float,
        default=0.98,
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=96,
    )
    parser.add_argument(
        '--n-heads',
        type=int,
        default=6,
    )
    parser.add_argument(
        '--transformer-layers',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--head-id-col',
        default=None,
    )
    parser.add_argument(
        '--tail-id-col',
        default=None,
    )
    parser.add_argument(
        '--head-smiles-col',
        default=None,
    )
    parser.add_argument(
        '--tail-smiles-col',
        default=None,
    )
    parser.add_argument(
        '--relation-col',
        default=None,
    )
    parser.add_argument(
        '--delimiter',
        default=None,
    )

    parser.add_argument(
        '--std-ddof',
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            '0: population standard deviation; '
            '1: sample standard deviation.'
        ),
    )

    return parser.parse_args()


def base_preprocess_command(
    args,
    operation,
    seed=None,
):
    command = [
        sys.executable,
        'data_pre.py',
        '--dataset',
        args.dataset,
        '--raw-file',
        args.raw_file,
        '--output-root',
        args.output_root,
        '--operation',
        operation,
        '--sim-topk',
        str(args.sim_topk),
        '--sim-quantile',
        str(args.sim_quantile),
        '--sim-std-lambda',
        str(args.sim_std_lambda),
        '--d-min',
        str(args.d_min),
        '--d-max',
        str(args.d_max),
    ]

    append_optional_argument(
        command,
        '--head-id-col',
        args.head_id_col,
    )
    append_optional_argument(
        command,
        '--tail-id-col',
        args.tail_id_col,
    )
    append_optional_argument(
        command,
        '--head-smiles-col',
        args.head_smiles_col,
    )
    append_optional_argument(
        command,
        '--tail-smiles-col',
        args.tail_smiles_col,
    )
    append_optional_argument(
        command,
        '--relation-col',
        args.relation_col,
    )
    append_optional_argument(
        command,
        '--delimiter',
        args.delimiter,
    )

    if operation == 'split':
        command.extend(
            [
                '--mode',
                args.mode,
                '--seed',
                str(seed),
                '--unseen-ratio',
                str(args.unseen_ratio),
            ]
        )

    return command


def train_command(args, seed):
    return [
        sys.executable,
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
        '--epochs',
        str(args.epochs),
        '--batch-size',
        str(args.batch_size),
        '--weight-decay',
        str(args.weight_decay),
        '--dropout',
        str(args.dropout),
        '--lr-gamma',
        str(args.lr_gamma),
        '--hidden-dim',
        str(args.hidden_dim),
        '--n-heads',
        str(args.n_heads),
        '--transformer-layers',
        str(args.transformer_layers),
        '--num-workers',
        str(args.num_workers),
    ]


def load_seed_result(
    result_root,
    dataset,
    mode,
    seed,
):
    result_file = (
        Path(result_root)
        / dataset
        / mode
        / f'seed{seed}.json'
    )

    if not result_file.exists():
        raise FileNotFoundError(
            result_file
        )

    with result_file.open(
        'r',
        encoding='utf-8',
    ) as file:
        return json.load(file)


def aggregate_results(
    seed_results,
    mode,
    ddof,
):
    scenarios = (
        ['test']
        if mode == 'transductive'
        else ['s1', 's2']
    )

    summary = {}

    for scenario in scenarios:
        summary[scenario] = {}

        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    result[scenario][metric_name]
                    for result in seed_results
                ],
                dtype=np.float64,
            )

            summary[scenario][metric_name] = {
                'mean': float(
                    np.nanmean(values)
                ),
                'std': float(
                    np.nanstd(
                        values,
                        ddof=ddof,
                    )
                ),
                'values': values.tolist(),
            }

    return summary


def main():
    args = parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(
            '--seeds contains duplicate values.'
        )

    if not args.skip_drug_data:
        run_command(
            base_preprocess_command(
                args,
                operation='drug_data',
            )
        )

    seed_results = []

    for seed in args.seeds:
        run_command(
            base_preprocess_command(
                args,
                operation='split',
                seed=seed,
            )
        )

        run_command(
            train_command(
                args,
                seed,
            )
        )

        seed_results.append(
            load_seed_result(
                result_root=args.result_root,
                dataset=args.dataset,
                mode=args.mode,
                seed=seed,
            )
        )

    summary = aggregate_results(
        seed_results=seed_results,
        mode=args.mode,
        ddof=args.std_ddof,
    )

    output = {
        'dataset': args.dataset,
        'mode': args.mode,
        'seeds': args.seeds,
        'std_ddof': args.std_ddof,
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

    for scenario, scenario_result in summary.items():
        print(f'\n[{scenario.upper()}]')

        for metric_name, metric_result in (
            scenario_result.items()
        ):
            print(
                f'{metric_name:10s}: '
                f'{metric_result["mean"]:.6f} '
                f'± {metric_result["std"]:.6f}'
            )

    print(
        f'\nSummary saved to {summary_file}'
    )


if __name__ == '__main__':
    main()
