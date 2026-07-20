import os
import pickle
from pathlib import Path

import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from utils import seed_worker


REQUIRED_COLUMNS = {
    'Drug1_ID',
    'Drug2_ID',
    'Y',
    'label',
}


def read_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class DrugDataset(Dataset):
    def __init__(
        self,
        data_df,
        drug_graph,
        drug_graph_dgl,
    ):
        missing_columns = REQUIRED_COLUMNS - set(data_df.columns)
        if missing_columns:
            raise ValueError(
                f'Dataset is missing columns: {missing_columns}'
            )

        self.data_df = data_df.reset_index(drop=True).copy()

     
        self.data_df['Drug1_ID'] = (
            self.data_df['Drug1_ID']
            .astype(str)
            .str.strip()
        )
        self.data_df['Drug2_ID'] = (
            self.data_df['Drug2_ID']
            .astype(str)
            .str.strip()
        )
        self.data_df['Y'] = (
            self.data_df['Y'].astype('int64')
        )
        self.data_df['label'] = (
            self.data_df['label'].astype('float32')
        )

       
        self.drug_graph = {
            str(drug_id).strip(): graph
            for drug_id, graph in drug_graph.items()
        }
        self.drug_graph_dgl = {
            str(drug_id).strip(): graph
            for drug_id, graph in drug_graph_dgl.items()
        }

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]

        return {
            'Drug1_ID': row['Drug1_ID'],
            'Drug2_ID': row['Drug2_ID'],
            'Y': int(row['Y']),
            'label': float(row['label']),
        }

    def collate_fn(self, batch):
        head_graphs = []
        tail_graphs = []
        head_dgl_graphs = []
        tail_dgl_graphs = []
        relations = []
        labels = []

        for row in batch:
            head_id = str(row['Drug1_ID'])
            tail_id = str(row['Drug2_ID'])

            head_graph = self.drug_graph.get(head_id)
            tail_graph = self.drug_graph.get(tail_id)

            head_dgl_graph = self.drug_graph_dgl.get(head_id)
            tail_dgl_graph = self.drug_graph_dgl.get(tail_id)

            if head_graph is None:
                raise KeyError(
                    f'No PyG molecular graph for drug {head_id}.'
                )
            if tail_graph is None:
                raise KeyError(
                    f'No PyG molecular graph for drug {tail_id}.'
                )
            if head_dgl_graph is None:
                raise KeyError(
                    f'No DGL molecular graph for drug {head_id}.'
                )
            if tail_dgl_graph is None:
                raise KeyError(
                    f'No DGL molecular graph for drug {tail_id}.'
                )

            head_graphs.append(head_graph)
            tail_graphs.append(tail_graph)
            head_dgl_graphs.append(head_dgl_graph)
            tail_dgl_graphs.append(tail_dgl_graph)

            relations.append(int(row['Y']))
            labels.append(float(row['label']))

        head_batch = Batch.from_data_list(
            head_graphs,
            follow_batch=['edge_index'],
        )
        tail_batch = Batch.from_data_list(
            tail_graphs,
            follow_batch=['edge_index'],
        )

        head_dgl_batch = dgl.batch(head_dgl_graphs)
        tail_dgl_batch = dgl.batch(tail_dgl_graphs)

        relation_tensor = torch.tensor(
            relations,
            dtype=torch.long,
        )
        label_tensor = torch.tensor(
            labels,
            dtype=torch.float32,
        )

        return (
            head_batch,
            tail_batch,
            head_dgl_batch,
            tail_dgl_batch,
            relation_tensor,
            label_tensor,
        )


class DrugDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(
            dataset,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


def build_loader(
    dataframe,
    drug_graph,
    drug_graph_dgl,
    batch_size,
    shuffle,
    num_workers,
    seed,
):
    dataset = DrugDataset(
        data_df=dataframe,
        drug_graph=drug_graph,
        drug_graph_dgl=drug_graph_dgl,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DrugDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def load_ddi_dataset(
    root,
    batch_size,
    mode,
    seed,
    num_workers=0,
):
  
    root = Path(root)

    drug_graph_file = root / 'drug_data_pyg.pkl'
    drug_graph_dgl_file = root / 'drug_data_dgl.pkl'

    if not drug_graph_file.exists():
        raise FileNotFoundError(drug_graph_file)
    if not drug_graph_dgl_file.exists():
        raise FileNotFoundError(drug_graph_dgl_file)

    drug_graph = read_pickle(drug_graph_file)
    drug_graph_dgl = read_pickle(drug_graph_dgl_file)

    if mode == 'transductive':
        split_names = ['train', 'val', 'test']
    elif mode == 'inductive':
        split_names = ['train', 'val', 's1', 's2']
    else:
        raise ValueError(
            f'Unsupported evaluation mode: {mode}'
        )

    loaders = {}

    for split_name in split_names:
        csv_file = (
            root
            / f'{mode}_seed{seed}_{split_name}.csv'
        )
        if not csv_file.exists():
            raise FileNotFoundError(
                f'{csv_file} does not exist. '
                'Generate this split with data_pre.py first.'
            )

        dataframe = pd.read_csv(
            csv_file,
            dtype={
                'Drug1_ID': str,
                'Drug2_ID': str,
            },
        )

        loaders[split_name] = build_loader(
            dataframe=dataframe,
            drug_graph=drug_graph,
            drug_graph_dgl=drug_graph_dgl,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            seed=seed,
        )

        print(
            f'{mode}/{split_name}: '
            f'{len(loaders[split_name].dataset)} instances'
        )

    return loaders
