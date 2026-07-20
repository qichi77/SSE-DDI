import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    'Drug1_ID',
    'Drug2_ID',
    'Y',
    'label',
}


def load_split(
    root,
    mode,
    seed,
    split_name,
):
    filename = (
        root
        / f'{mode}_seed{seed}_{split_name}.csv'
    )

    dataframe = pd.read_csv(
        filename,
        dtype={
            'Drug1_ID': str,
            'Drug2_ID': str,
        },
    )

    missing = REQUIRED_COLUMNS - set(
        dataframe.columns
    )
    if missing:
        raise AssertionError(
            f'{filename} missing columns: {missing}'
        )

    dataframe['Drug1_ID'] = (
        dataframe['Drug1_ID']
        .astype(str)
        .str.strip()
    )
    dataframe['Drug2_ID'] = (
        dataframe['Drug2_ID']
        .astype(str)
        .str.strip()
    )
    dataframe['Y'] = (
        dataframe['Y'].astype(int)
    )
    dataframe['label'] = (
        dataframe['label'].astype(int)
    )

    return dataframe


def triplet_set(dataframe):
    return set(
        zip(
            dataframe['Drug1_ID'],
            dataframe['Drug2_ID'],
            dataframe['Y'],
        )
    )


def validate_binary_split(
    split_name,
    dataframe,
):
    if dataframe.empty:
        raise AssertionError(
            f'{split_name} is empty.'
        )

    invalid_labels = (
        set(dataframe['label'].unique())
        - {0, 1}
    )
    if invalid_labels:
        raise AssertionError(
            f'{split_name} invalid labels: '
            f'{invalid_labels}'
        )

    positives = dataframe.loc[
        dataframe['label'] == 1
    ]
    negatives = dataframe.loc[
        dataframe['label'] == 0
    ]

    if len(positives) != len(negatives):
        raise AssertionError(
            f'{split_name} is not 1:1: '
            f'{len(positives)} positives, '
            f'{len(negatives)} negatives.'
        )

    if positives.duplicated(
        subset=[
            'Drug1_ID',
            'Drug2_ID',
            'Y',
        ]
    ).any():
        raise AssertionError(
            f'{split_name} has duplicate positives.'
        )

    if negatives.duplicated(
        subset=[
            'Drug1_ID',
            'Drug2_ID',
            'Y',
        ]
    ).any():
        raise AssertionError(
            f'{split_name} has duplicate negatives.'
        )

    return positives, negatives


def validate_global_sets(
    positive_sets,
    negative_sets,
):
    split_names = list(positive_sets)

    for index, first_name in enumerate(
        split_names
    ):
        for second_name in split_names[
            index + 1:
        ]:
            overlap = (
                positive_sets[first_name]
                & positive_sets[second_name]
            )

            if overlap:
                raise AssertionError(
                    f'Positive overlap between '
                    f'{first_name} and {second_name}: '
                    f'{len(overlap)}'
                )

    all_positives = set().union(
        *positive_sets.values()
    )
    all_negatives = set().union(
        *negative_sets.values()
    )

    positive_negative_overlap = (
        all_positives & all_negatives
    )
    if positive_negative_overlap:
        raise AssertionError(
            'Generated negatives overlap observed '
            f'positives: {len(positive_negative_overlap)}'
        )

    total_negative_count = sum(
        len(values)
        for values in negative_sets.values()
    )

    if len(all_negatives) != total_negative_count:
        raise AssertionError(
            'Duplicate negatives exist across splits.'
        )


def validate_transductive(
    root,
    seed,
):
    split_names = [
        'train',
        'val',
        'test',
    ]

    frames = {
        name: load_split(
            root,
            'transductive',
            seed,
            name,
        )
        for name in split_names
    }

    positive_sets = {}
    negative_sets = {}
    positive_counts = {}

    for name, dataframe in frames.items():
        positives, negatives = (
            validate_binary_split(
                name,
                dataframe,
            )
        )

        positive_sets[name] = triplet_set(
            positives
        )
        negative_sets[name] = triplet_set(
            negatives
        )
        positive_counts[name] = len(
            positives
        )

    validate_global_sets(
        positive_sets,
        negative_sets,
    )

    total = sum(positive_counts.values())

    observed_ratios = np.asarray(
        [
            positive_counts['train'] / total,
            positive_counts['val'] / total,
            positive_counts['test'] / total,
        ]
    )
    expected_ratios = np.asarray(
        [0.60, 0.20, 0.20]
    )

    tolerance = max(
        1.0 / total,
        0.005,
    )

    if not np.allclose(
        observed_ratios,
        expected_ratios,
        atol=tolerance,
    ):
        raise AssertionError(
            'Unexpected transductive ratios: '
            f'{observed_ratios.tolist()}'
        )


def validate_inductive(
    root,
    seed,
):
    split_names = [
        'train',
        'val',
        's1',
        's2',
    ]

    frames = {
        name: load_split(
            root,
            'inductive',
            seed,
            name,
        )
        for name in split_names
    }

    unseen_file = (
        root
        / f'inductive_seed{seed}_unseen_drugs.txt'
    )

    unseen_drugs = {
        line.strip()
        for line in unseen_file.read_text(
            encoding='utf-8'
        ).splitlines()
        if line.strip()
    }

    if not unseen_drugs:
        raise AssertionError(
            'Unseen-drug file is empty.'
        )

    positive_sets = {}
    negative_sets = {}

    for name, dataframe in frames.items():
        positives, negatives = (
            validate_binary_split(
                name,
                dataframe,
            )
        )

        positive_sets[name] = triplet_set(
            positives
        )
        negative_sets[name] = triplet_set(
            negatives
        )

        head_unseen = (
            dataframe['Drug1_ID']
            .isin(unseen_drugs)
        )
        tail_unseen = (
            dataframe['Drug2_ID']
            .isin(unseen_drugs)
        )

        if name in {'train', 'val'}:
            valid = (
                ~head_unseen
                & ~tail_unseen
            )
        elif name == 's1':
            valid = (
                head_unseen
                ^ tail_unseen
            )
        else:
            valid = (
                head_unseen
                & tail_unseen
            )

        if not bool(valid.all()):
            raise AssertionError(
                f'{name} violates its '
                'seen/unseen definition.'
            )

    validate_global_sets(
        positive_sets,
        negative_sets,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
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
        '--seed',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--data-root',
        default='./data/processed',
    )

    args = parser.parse_args()

    root = (
        Path(args.data_root)
        / args.dataset
    )

    if args.mode == 'transductive':
        validate_transductive(
            root,
            args.seed,
        )
    else:
        validate_inductive(
            root,
            args.seed,
        )

    print(
        f'Validation passed: '
        f'{args.dataset}/'
        f'{args.mode}/seed{args.seed}'
    )


if __name__ == '__main__':
    main()
