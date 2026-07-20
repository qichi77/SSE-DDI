import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from data_pre import CustomData  
from dataset import load_ddi_dataset
from metrics import do_compute_metrics
from model import gnn_model
from utils import AverageMeter, set_seed


METRIC_NAMES = [
    'acc',
    'auc',
    'f1',
    'precision',
    'recall',
    'ap',
]


def move_batch_to_device(batch, device):
    return tuple(
        item.to(device)
        for item in batch
    )


def forward_batch(model, batch, device):
    (
        head_pairs,
        tail_pairs,
        head_pairs_dgl,
        tail_pairs_dgl,
        relation,
        label,
    ) = move_batch_to_device(batch, device)

    head_similarity = head_pairs.sim
    tail_similarity = tail_pairs.sim

    head_edge_features = (
        head_pairs_dgl.edata['feat']
    )
    tail_edge_features = (
        tail_pairs_dgl.edata['feat']
    )

    logits = model(
        head_pairs,
        tail_pairs,
        head_pairs_dgl,
        tail_pairs_dgl,
        head_edge_features,
        tail_edge_features,
        relation,
        head_similarity,
        tail_similarity,
    )

    return logits.view(-1), label.view(-1)


def compute_metric_dict(probabilities, labels):
    metric_values = do_compute_metrics(
        probabilities,
        labels,
    )

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in zip(
            METRIC_NAMES,
            metric_values,
        )
    }


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
):
    model.train()

    loss_meter = AverageMeter()
    all_probabilities = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc=f'train epoch {epoch}',
        leave=False,
    )

    for batch in progress_bar:
        logits, labels = forward_batch(
            model=model,
            batch=batch,
            device=device,
        )

        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits)

        loss_meter.update(
            loss.item(),
            labels.size(0),
        )

        all_probabilities.append(
            probabilities.detach().cpu().numpy()
        )
        all_labels.append(
            labels.detach().cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f'{loss_meter.get_average():.4f}'
        )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )
    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    result = compute_metric_dict(
        probabilities,
        labels,
    )
    result['loss'] = float(
        loss_meter.get_average()
    )

    return result


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    split_name,
):
    model.eval()

    loss_meter = AverageMeter()
    all_probabilities = []
    all_labels = []

    for batch in tqdm(
        dataloader,
        desc=f'evaluate {split_name}',
        leave=False,
    ):
        logits, labels = forward_batch(
            model=model,
            batch=batch,
            device=device,
        )

        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        loss_meter.update(
            loss.item(),
            labels.size(0),
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )
        all_labels.append(
            labels.cpu().numpy()
        )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )
    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    result = compute_metric_dict(
        probabilities,
        labels,
    )
    result['loss'] = float(
        loss_meter.get_average()
    )

    return result


def format_metrics(metrics):
    fields = [
        f'{name}={value:.4f}'
        for name, value in metrics.items()
    ]
    return ', '.join(fields)


def save_json(data, output_file):
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train paper-consistent SSE-DDI.'
    )

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
        '--seed',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--data-root',
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
        '--learning-rate',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

   
    if args.mode == 'transductive':
        learning_rate = (
            1e-4
            if args.learning_rate is None
            else args.learning_rate
        )
        n_iter = (
            8
            if args.n_iter is None
            else args.n_iter
        )
    else:
        learning_rate = (
            2e-5
            if args.learning_rate is None
            else args.learning_rate
        )
        n_iter = (
            6
            if args.n_iter is None
            else args.n_iter
        )

    dataset_root = (
        Path(args.data_root) / args.dataset
    )

    metadata_file = (
        dataset_root / 'dataset_meta.json'
    )
    if not metadata_file.exists():
        raise FileNotFoundError(
            f'{metadata_file} does not exist.'
        )

    with metadata_file.open(
        'r',
        encoding='utf-8',
    ) as file:
        metadata = json.load(file)

    loaders = load_ddi_dataset(
        root=dataset_root,
        batch_size=args.batch_size,
        mode=args.mode,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    first_batch = next(iter(loaders['train']))

    node_dim = int(
        first_batch[0].x.size(-1)
    )
    edge_dim = int(
        first_batch[0].edge_attr.size(-1)
    )
    observed_similarity_dim = int(
        first_batch[0].sim.size(-1)
    )

    metadata_similarity_dim = int(
        metadata['similarity_dim']
    )
    if (
        observed_similarity_dim
        != metadata_similarity_dim
    ):
        raise ValueError(
            'Similarity input dimension mismatch: '
            f'batch={observed_similarity_dim}, '
            f'metadata={metadata_similarity_dim}'
        )

    if args.hidden_dim % args.n_heads != 0:
        raise ValueError(
            '--hidden-dim must be divisible by --n-heads.'
        )

    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )

    net_params = {
        'L': args.transformer_layers,
        'n_heads': args.n_heads,
        'hidden_dim': args.hidden_dim,
        'out_dim': args.hidden_dim,
        'edge_feat': True,
        'residual': True,
        'readout': 'mean',
        'in_feat_dropout': args.dropout,
        'dropout': args.dropout,
        'layer_norm': False,
        'batch_norm': True,
        'self_loop': False,
        'lap_pos_enc': True,
        'pos_enc_dim': 6,
        'full_graph': False,
        'batch_size': args.batch_size,
        'num_atom_type': node_dim,
        'num_bond_type': edge_dim,
        'device': device,
        'n_iter': n_iter,
        'num_relations': int(
            metadata['num_relations']
        ),
        'similarity_dim': metadata_similarity_dim,
    }

    model = gnn_model(
        'GraphTransformer',
        net_params,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.lr_gamma,
    )

    best_val_auc = -math.inf
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = float(
            optimizer.param_groups[0]['lr']
        )
        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=loaders['val'],
            criterion=criterion,
            device=device,
            split_name='validation',
        )

        current_val_auc = val_metrics['auc']

       
        if math.isnan(current_val_auc):
            selection_score = -val_metrics['loss']
        else:
            selection_score = current_val_auc

        if selection_score > best_val_auc:
            best_val_auc = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        scheduler.step()

        epoch_record = {
            'epoch': epoch,
            'learning_rate': epoch_learning_rate,
            'train': train_metrics,
            'validation': val_metrics,
        }
        history.append(epoch_record)

        print(
            f'Epoch {epoch:03d} | '
            f'train: {format_metrics(train_metrics)} | '
            f'validation: {format_metrics(val_metrics)}'
        )

    if best_state is None:
        raise RuntimeError(
            'No valid model checkpoint was selected.'
        )

    model.load_state_dict(best_state)

    checkpoint_dir = (
        Path(args.checkpoint_root)
        / args.dataset
        / args.mode
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_file = (
        checkpoint_dir
        / f'seed{args.seed}_best.pt'
    )
    torch.save(best_state, checkpoint_file)

    result = {
        'dataset': args.dataset,
        'mode': args.mode,
        'seed': int(args.seed),
        'best_epoch': int(best_epoch),
        'checkpoint': str(checkpoint_file),
        'hyperparameters': {
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'learning_rate': float(learning_rate),
            'weight_decay': float(args.weight_decay),
            'lr_gamma': float(args.lr_gamma),
            'dropout': float(args.dropout),
            'hidden_dim': int(args.hidden_dim),
            'n_heads': int(args.n_heads),
            'transformer_layers': int(
                args.transformer_layers
            ),
            'n_iter': int(n_iter),
        },
        'history': history,
    }

    if args.mode == 'transductive':
        result['test'] = evaluate(
            model=model,
            dataloader=loaders['test'],
            criterion=criterion,
            device=device,
            split_name='test',
        )
        print(
            'Test: '
            + format_metrics(result['test'])
        )
    else:
        result['s1'] = evaluate(
            model=model,
            dataloader=loaders['s1'],
            criterion=criterion,
            device=device,
            split_name='S1',
        )
        result['s2'] = evaluate(
            model=model,
            dataloader=loaders['s2'],
            criterion=criterion,
            device=device,
            split_name='S2',
        )

        print(
            'S1: '
            + format_metrics(result['s1'])
        )
        print(
            'S2: '
            + format_metrics(result['s2'])
        )

    result_file = (
        Path(args.result_root)
        / args.dataset
        / args.mode
        / f'seed{args.seed}.json'
    )
    save_json(result, result_file)

    print(f'Result saved to {result_file}')


if __name__ == '__main__':
    main()
