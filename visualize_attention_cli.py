import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Batch

from data_pre import CustomData  # noqa: F401
from dataset import load_ddi_dataset
from model import gnn_model


def load_checkpoint(
    checkpoint_file,
    net_params,
    device,
):
    payload = torch.load(
        checkpoint_file,
        map_location=device,
    )

    if (
        isinstance(payload, dict)
        and 'model_state_dict' in payload
    ):
        state_dict = payload[
            'model_state_dict'
        ]
    else:
        # 兼容旧 checkpoint。
        state_dict = payload

    model = gnn_model(
        'GraphTransformer',
        net_params,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def minmax_normalize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    value_min = float(values.min())
    value_max = float(values.max())

    if value_max - value_min <= 1e-12:
        return np.zeros_like(values)

    return (
        (values - value_min)
        / (value_max - value_min)
    )


def extract_final_pool_weights(
    model,
    graph_batch,
):
    """
    DMPNN 在每次 SSE 迭代中覆盖 att_weights。
    编码完成后保留的是最后一次迭代的 pooling weights。
    """
    graph_copy = graph_batch.clone()

    _ = model.drug_encoder(
        graph_copy
    )

    attention_module = (
        model
        .drug_encoder
        .line_graph
        .att
    )

    weights = getattr(
        attention_module,
        'att_weights',
        None,
    )

    if weights is None:
        raise RuntimeError(
            'The SSE encoder did not expose '
            'final pooling weights.'
        )

    return (
        weights
        .detach()
        .reshape(-1)
    )


def bond_state_to_atom_scores(
    graph_batch,
    bond_state_weights,
):
    """
    对每个原子，将所有 incident directed bond-state
    的权重取平均，再进行分子内部 min-max normalization。
    """
    atom_score_list = []

    for graph_index in range(
        graph_batch.num_graphs
    ):
        node_start = int(
            graph_batch.ptr[graph_index]
        )
        node_end = int(
            graph_batch.ptr[
                graph_index + 1
            ]
        )
        num_atoms = node_end - node_start

        edge_mask = (
            graph_batch.edge_index_batch
            == graph_index
        )

        local_edge_index = (
            graph_batch.edge_index[
                :,
                edge_mask,
            ]
            - node_start
        )

        local_weights = (
            bond_state_weights[
                edge_mask
            ]
        )

        score_sum = torch.zeros(
            num_atoms,
            device=local_weights.device,
            dtype=local_weights.dtype,
        )
        score_count = torch.zeros_like(
            score_sum
        )

        if local_edge_index.numel() > 0:
            source = local_edge_index[0]
            target = local_edge_index[1]

            score_sum.index_add_(
                0,
                source,
                local_weights,
            )
            score_sum.index_add_(
                0,
                target,
                local_weights,
            )

            ones = torch.ones_like(
                local_weights
            )
            score_count.index_add_(
                0,
                source,
                ones,
            )
            score_count.index_add_(
                0,
                target,
                ones,
            )

        atom_scores = (
            score_sum
            / score_count.clamp_min(1.0)
        )

        normalized = minmax_normalize(
            atom_scores.cpu().numpy()
        )

        atom_score_list.append(
            normalized
        )

    return atom_score_list


def draw_atom_attention(
    smiles,
    atom_scores,
    output_file,
):
    molecule = Chem.MolFromSmiles(
        smiles
    )
    if molecule is None:
        raise ValueError(
            f'Invalid SMILES: {smiles}'
        )

    if (
        molecule.GetNumAtoms()
        != len(atom_scores)
    ):
        raise ValueError(
            'SMILES atom count does not match '
            'the graph atom-score count.'
        )

    highlight_atoms = list(
        range(molecule.GetNumAtoms())
    )

    # 低分接近白色，高分接近红色。
    highlight_colors = {
        atom_index: (
            1.0,
            float(1.0 - score),
            float(1.0 - score),
        )
        for atom_index, score in enumerate(
            atom_scores
        )
    }

    highlight_radii = {
        atom_index: (
            0.25 + 0.25 * float(score)
        )
        for atom_index, score in enumerate(
            atom_scores
        )
    }

    image = Draw.MolToImage(
        molecule,
        size=(600, 450),
        highlightAtoms=highlight_atoms,
        highlightAtomColors=(
            highlight_colors
        ),
        highlightAtomRadii=(
            highlight_radii
        ),
        legend='Atom-level SSE attention',
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    image.save(output_file)


def move_batch_to_device(
    batch,
    device,
):
    return tuple(
        item.to(device)
        for item in batch
    )


def build_net_params(
    metadata,
    first_batch,
    mode,
    device,
    batch_size,
):
    hidden_dim = 96
    n_heads = 6

    return {
        'L': 2,
        'n_heads': n_heads,
        'hidden_dim': hidden_dim,
        'out_dim': hidden_dim,
        'edge_feat': True,
        'residual': True,
        'readout': 'mean',
        'in_feat_dropout': 0.2,
        'dropout': 0.2,
        'layer_norm': False,
        'batch_norm': True,
        'self_loop': False,
        'lap_pos_enc': True,
        'pos_enc_dim': 6,
        'full_graph': False,
        'batch_size': batch_size,
        'num_atom_type': int(
            first_batch[0].x.size(-1)
        ),
        'num_bond_type': int(
            first_batch[0]
            .edge_attr
            .size(-1)
        ),
        'device': device,
        'n_iter': (
            8
            if mode == 'transductive'
            else 6
        ),
        'num_relations': int(
            metadata['num_relations']
        ),
        'similarity_dim': int(
            metadata['similarity_dim']
        ),
    }


def main():
    parser = argparse.ArgumentParser()

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
        '--split',
        choices=[
            'train',
            'val',
            'test',
            's1',
            's2',
        ],
        required=True,
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
    )
    parser.add_argument(
        '--data-root',
        default='./data/processed',
    )
    parser.add_argument(
        '--output-dir',
        default='./attention_output',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--correct-positive-only',
        action='store_true',
    )

    args = parser.parse_args()

    if (
        args.mode == 'transductive'
        and args.split in {'s1', 's2'}
    ):
        raise ValueError(
            'S1/S2 require inductive mode.'
        )

    if (
        args.mode == 'inductive'
        and args.split == 'test'
    ):
        raise ValueError(
            'Inductive mode uses S1 and S2.'
        )

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    dataset_root = (
        Path(args.data_root)
        / args.dataset
    )

    with (
        dataset_root
        / 'dataset_meta.json'
    ).open(
        'r',
        encoding='utf-8',
    ) as file:
        metadata = json.load(file)

    loaders = load_ddi_dataset(
        root=dataset_root,
        batch_size=args.batch_size,
        mode=args.mode,
        seed=args.seed,
        num_workers=0,
    )

    loader = loaders[args.split]
    first_batch = next(iter(loader))

    net_params = build_net_params(
        metadata=metadata,
        first_batch=first_batch,
        mode=args.mode,
        device=device,
        batch_size=args.batch_size,
    )

    model = load_checkpoint(
        checkpoint_file=args.checkpoint,
        net_params=net_params,
        device=device,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for batch in loader:
        (
            head,
            tail,
            head_dgl,
            tail_dgl,
            relation,
            labels,
        ) = move_batch_to_device(
            batch,
            device,
        )

        # 保存未被模型原地修改的图。
        head_original = head.clone()
        tail_original = tail.clone()

        with torch.no_grad():
            logits = model(
                head.clone(),
                tail.clone(),
                head_dgl,
                tail_dgl,
                head_dgl.edata['feat'],
                tail_dgl.edata['feat'],
                relation,
                head.sim,
                tail.sim,
            )

            probabilities = torch.sigmoid(
                logits
            )

            head_weights = (
                extract_final_pool_weights(
                    model,
                    head_original,
                )
            )
            tail_weights = (
                extract_final_pool_weights(
                    model,
                    tail_original,
                )
            )

        head_atom_scores = (
            bond_state_to_atom_scores(
                head_original,
                head_weights,
            )
        )
        tail_atom_scores = (
            bond_state_to_atom_scores(
                tail_original,
                tail_weights,
            )
        )

        head_graphs = (
            head_original.to_data_list()
        )
        tail_graphs = (
            tail_original.to_data_list()
        )

        for index in range(len(labels)):
            label = int(
                labels[index].item()
            )
            probability = float(
                probabilities[index].item()
            )

            if args.correct_positive_only:
                if not (
                    label == 1
                    and probability >= 0.50
                ):
                    continue

            head_smiles = (
                head_graphs[index].smiles
            )
            tail_smiles = (
                tail_graphs[index].smiles
            )

            prefix = (
                f'{args.split}_'
                f'sample{saved_count:04d}_'
                f'p{probability:.4f}'
            )

            draw_atom_attention(
                smiles=head_smiles,
                atom_scores=(
                    head_atom_scores[index]
                ),
                output_file=(
                    output_dir
                    / f'{prefix}_head.png'
                ),
            )

            draw_atom_attention(
                smiles=tail_smiles,
                atom_scores=(
                    tail_atom_scores[index]
                ),
                output_file=(
                    output_dir
                    / f'{prefix}_tail.png'
                ),
            )

            saved_count += 1

            if (
                saved_count
                >= args.num_samples
            ):
                print(
                    f'Saved {saved_count} samples '
                    f'to {output_dir}'
                )
                return

    print(
        f'Saved {saved_count} samples '
        f'to {output_dir}'
    )


if __name__ == '__main__':
    main()
