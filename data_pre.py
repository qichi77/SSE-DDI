from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import train_test_split
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

import torch
import pickle

import torch.utils.data
import os

import dgl

from scipy import sparse as sp
import numpy as np
import json
from pathlib import Path

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s,
                    allowable_set))

def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def generate_drug_data(
    mol_graph,
    atom_symbols,
    fps_all,
    id,
    self_idx=None,
    topk=32,
    tau_mode='p70',
    d_min=8
):

    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()]
    )
    if edge_list.numel() > 0:
        edge_list, edge_feats = edge_list[:, :2], edge_list[:, 2:].float()

        edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0)
        edge_feats = torch.cat([edge_feats] * 2, dim=0)
    else:
        edge_list = torch.empty((0, 2), dtype=torch.long)
        edge_feats = torch.empty((0, 6), dtype=torch.float32)

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    if len(features) == 0:
        raise ValueError("Molecule has no atoms; cannot create node features.")
    _, features = zip(*features)
    features = torch.stack(features)  # [num_nodes, feat_dim]

    if edge_list.numel() > 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & \
               (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T
        new_edge_index = edge_list.T  # [2, E]
    else:
        line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_index = torch.empty((2, 0), dtype=torch.long)


    N = len(fps_all)
    if self_idx is None or not (0 <= self_idx < N):

        from rdkit.Chem import AllChem
        mol_graph_fps = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)
    else:
        mol_graph_fps = fps_all[self_idx]

    similarity_vector = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        similarity_vector[i] = DataStructs.FingerprintSimilarity(fps_all[i], mol_graph_fps)


    if self_idx is not None and 0 <= self_idx < N:
        similarity_vector[self_idx] = 0.0


    if similarity_vector.numel() == 0:
        sparse_sim = similarity_vector.clone()
    else:
        k = min(topk, similarity_vector.numel())
        topk_values, topk_indices = torch.topk(similarity_vector, k)

        positive = similarity_vector[similarity_vector > 0]
        if tau_mode == 'p70':
            tau = torch.quantile(positive, 0.70) if positive.numel() > 0 else torch.tensor(0.0)
        elif tau_mode == 'mean+0.5std':
            tau = (positive.mean() + 0.5 * positive.std()) if positive.numel() > 0 else torch.tensor(0.0)
        else:
            tau = torch.tensor(float(tau_mode))

        mask = topk_values >= tau
        kept_indices = topk_indices[mask]
        kept_values  = topk_values[mask]


        if kept_indices.numel() < d_min:
            need = d_min - kept_indices.numel()
            cand_indices = topk_indices[~mask]
            cand_values  = topk_values[~mask]
            valid = cand_values > 0
            cand_indices = cand_indices[valid]
            cand_values  = cand_values[valid]
            if cand_indices.numel() > 0:
                order = torch.argsort(cand_values, descending=True)[:need]
                kept_indices = torch.cat([kept_indices, cand_indices[order]], dim=0)
                kept_values  = torch.cat([kept_values,  cand_values[order]],  dim=0)

        sparse_sim = torch.zeros_like(similarity_vector)
        if kept_indices.numel() > 0:
            sparse_sim[kept_indices] = kept_values


    data = CustomData(
        x=features,
        edge_index=new_edge_index,
        line_graph_edge_index=line_graph_edge_index,
        edge_attr=edge_feats,
        sim=sparse_sim.unsqueeze(0),  # [1, N]
        id=id
    )
    return data



def generate_drug_data_dgl(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)
    node_feature = features.long()
    edge_feature = edge_feats.long()

    g = dgl.DGLGraph()
    g.add_nodes(features.shape[0])
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature
    data_dgl = g
    return data_dgl


def finalize_similarity_graph(drug_data_pyg, id_to_idx, d_min=8, d_max=64, make_symmetric='min'):

    ids = list(drug_data_pyg.keys())
    N = len(ids)


    rows, cols, vals = [], [], []
    for id_ in ids:
        i = id_to_idx[id_]
        sim_row = drug_data_pyg[id_].sim.squeeze(0)  # [N]
        nz = torch.nonzero(sim_row > 0, as_tuple=False).flatten()
        if nz.numel() > 0:
            rows.append(torch.full((nz.numel(),), i, dtype=torch.long))
            cols.append(nz)
            vals.append(sim_row[nz])
    if len(rows) == 0:
        return drug_data_pyg

    rows = torch.cat(rows); cols = torch.cat(cols); vals = torch.cat(vals)
    S = sp.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(N, N)).tocsr()


    S_T = S.transpose().tocsr()
    if make_symmetric == 'min':
        S_mut = S.minimum(S_T)
        S_mut.eliminate_zeros()  # 清理显式0
    elif make_symmetric == 'mean':

        M = S.minimum(S_T)
        M.eliminate_zeros()
        S_mut = M
    else:
        raise ValueError('make_symmetric must be "min" or "mean"')


    def row_topk_csr(A, k):
        A = A.tolil()
        for i in range(A.shape[0]):
            row_data = A.data[i]; row_idx = A.rows[i]
            if len(row_data) > k:
                order = np.argsort(row_data)[::-1][:k]
                A.data[i] = list(np.array(row_data)[order])
                A.rows[i] = list(np.array(row_idx)[order])
        return A.tocsr()

    S_mut = row_topk_csr(S_mut, d_max)


    degrees = np.diff(S_mut.indptr)
    need_fill = np.where(degrees < d_min)[0]
    if need_fill.size > 0:
        S_mut = S_mut.tolil()
        S_orig = S.tocsr()
        for i in need_fill:
            have_set = set(S_mut.rows[i])
            row = S_orig.getrow(i)  # 原始单向 TopK∧阈值结果
            if row.nnz == 0:
                continue
            order = np.argsort(row.data)[::-1]
            for idx in order:
                j = row.indices[idx]
                if j not in have_set and i != j and row.data[idx] > 0:
                    S_mut.rows[i].append(j)
                    S_mut.data[i].append(row.data[idx])
                    have_set.add(j)
                    if len(S_mut.rows[i]) >= d_min:
                        break
        S_mut = S_mut.tocsr()



    S_mut = S_mut.tocsr()
    for id_ in ids:
        i = id_to_idx[id_]
        row = S_mut.getrow(i)
        vec = torch.zeros(N, dtype=torch.float32)
        if row.nnz > 0:
            vec[row.indices] = torch.from_numpy(row.data).float()
        drug_data_pyg[id_].sim = vec.unsqueeze(0)

    return drug_data_pyg


def load_drug_mol_data(
    args,
    topk=32,
    tau_mode='p70',
    d_min=8,
    d_max=64,
    do_finalize=True
):

    df = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    needed_cols = [args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y]
    df = df[needed_cols].copy()
    df[args.c_id1] = df[args.c_id1].astype(str).str.strip()
    df[args.c_id2] = df[args.c_id2].astype(str).str.strip()


    drug_smile_dict = {}
    for id1, id2, smi1, smi2, _ in zip(df[args.c_id1], df[args.c_id2], df[args.c_s1], df[args.c_s2], df[args.c_y]):
        if id1 not in drug_smile_dict:
            drug_smile_dict[id1] = smi1
        if id2 not in drug_smile_dict:
            drug_smile_dict[id2] = smi2


    drug_id_mol_tup = []   # [(id, mol)]
    symbols = []
    for did, smi in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            continue
        drug_id_mol_tup.append((did, mol))
        symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
    symbols = list(set(symbols))


    drug_id_mol_tup.sort(key=lambda x: str(x[0]))


    id_to_idx = {did: idx for idx, (did, _) in enumerate(drug_id_mol_tup)}
    fps_all = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for _, mol in drug_id_mol_tup]


    drug_data_pyg = {}
    for did, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_pyg'):
        self_idx = id_to_idx[did]
        data_i = generate_drug_data(
            mol_graph=mol,
            atom_symbols=symbols,
            fps_all=fps_all,
            id=did,
            self_idx=self_idx,
            topk=topk,
            tau_mode=tau_mode,
            d_min=d_min
        )
        drug_data_pyg[did] = data_i


    drug_data_dgl = {did: generate_drug_data_dgl(mol, symbols)
                     for did, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}


    if do_finalize and 'finalize_similarity_graph' in globals():
        drug_data_pyg = finalize_similarity_graph(
            drug_data_pyg, id_to_idx, d_min=d_min, d_max=d_max, make_symmetric='min'
        )


    save_data(drug_data_pyg, 'drug_data_pyg.pkl', args)
    save_data(drug_data_dgl, 'drug_data_dgl.pkl', args)

    return drug_data_pyg, drug_data_dgl

def dataset_dir(args):

    path = Path(args.dirname) / args.dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_data(data, filename, args):
  
    output_file = dataset_dir(args) / filename
    with output_file.open('wb') as file:
        pickle.dump(data, file)

    print(f'\nData saved as {output_file}!')


def save_json(data, filename, args):
    output_file = dataset_dir(args) / filename
    with output_file.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f'Data saved as {output_file}!')


def load_valid_positive_triplets(args):

    graph_file = dataset_dir(args) / 'drug_data_pyg.pkl'

    if not graph_file.exists():
        raise FileNotFoundError(
            f'{graph_file} does not exist. '
            'Run data_pre.py with --operation drug_data first.'
        )

    with graph_file.open('rb') as file:
        drug_graph = pickle.load(file)


    valid_drug_ids = {str(drug_id).strip() for drug_id in drug_graph.keys()}
    similarity_dim = len(drug_graph)

    raw_df = pd.read_csv(
        args.dataset_filename,
        delimiter=args.delimiter,
    )

    required_columns = [
        args.c_id1,
        args.c_id2,
        args.c_y,
    ]
    missing_columns = [
        column for column in required_columns
        if column not in raw_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f'Missing columns in {args.dataset_filename}: '
            f'{missing_columns}. Existing columns: '
            f'{list(raw_df.columns)}'
        )

    raw_df = raw_df[required_columns].copy()
    raw_df = raw_df.dropna(subset=required_columns)

    raw_df[args.c_id1] = (
        raw_df[args.c_id1].astype(str).str.strip()
    )
    raw_df[args.c_id2] = (
        raw_df[args.c_id2].astype(str).str.strip()
    )


    valid_mask = (
        raw_df[args.c_id1].isin(valid_drug_ids)
        & raw_df[args.c_id2].isin(valid_drug_ids)
    )
    raw_df = raw_df.loc[valid_mask].copy()

    if raw_df.empty:
        raise ValueError(
            'No valid positive triplets remain after molecular-graph '
            'filtering. Check drug IDs, SMILES and column names.'
        )


   
    relation_values = sorted(
        raw_df[args.c_y].unique().tolist(),
        key=lambda value: str(value),
    )
    relation_map = {
        relation_value: relation_id
        for relation_id, relation_value in enumerate(relation_values)
    }

    positive_df = pd.DataFrame({
        'Drug1_ID': raw_df[args.c_id1],
        'Drug2_ID': raw_df[args.c_id2],
        'Y': raw_df[args.c_y].map(relation_map).astype(np.int64),
    })

  
    positive_df = (
        positive_df
        .drop_duplicates(
            subset=['Drug1_ID', 'Drug2_ID', 'Y'],
            keep='first',
        )
        .reset_index(drop=True)
    )

    candidate_drugs = sorted(
        set(positive_df['Drug1_ID'])
        | set(positive_df['Drug2_ID'])
    )

    serializable_relation_map = {
        str(raw_relation): int(encoded_relation)
        for raw_relation, encoded_relation in relation_map.items()
    }
    save_json(
        {
            'raw_to_id': serializable_relation_map,
            'num_relations': len(relation_map),
        },
        'relation_map.json',
        args,
    )

    return (
        positive_df,
        candidate_drugs,
        similarity_dim,
        serializable_relation_map,
    )


def split_transductive_positive_triplets(positive_df, seed):

    train_df, temporary_df = train_test_split(
        positive_df,
        test_size=0.40,
        random_state=seed,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_inductive_positive_triplets(
    positive_df,
    candidate_drugs,
    unseen_ratio,
    seed,
):

    rng = np.random.default_rng(seed)

    candidate_drugs_array = np.asarray(
        candidate_drugs,
        dtype=object,
    )
    num_unseen = int(
        round(len(candidate_drugs_array) * unseen_ratio)
    )
    num_unseen = max(1, num_unseen)
    num_unseen = min(
        num_unseen,
        len(candidate_drugs_array) - 1,
    )

    unseen_drugs = set(
        rng.choice(
            candidate_drugs_array,
            size=num_unseen,
            replace=False,
        ).tolist()
    )
    seen_drugs = set(candidate_drugs) - unseen_drugs

    head_unseen = positive_df['Drug1_ID'].isin(unseen_drugs)
    tail_unseen = positive_df['Drug2_ID'].isin(unseen_drugs)

    seen_seen_df = positive_df.loc[
        ~head_unseen & ~tail_unseen
    ].copy()

    s1_df = positive_df.loc[
        head_unseen ^ tail_unseen
    ].copy()

    s2_df = positive_df.loc[
        head_unseen & tail_unseen
    ].copy()

    if seen_seen_df.empty:
        raise ValueError(
            'The selected unseen-drug split produces no seen-seen '
            'training triplets.'
        )
    if s1_df.empty:
        raise ValueError(
            'The selected unseen-drug split produces no S1 triplets. '
            'Use another random seed.'
        )
    if s2_df.empty:
        raise ValueError(
            'The selected unseen-drug split produces no S2 triplets. '
            'Use another random seed.'
        )

    train_df, val_df = train_test_split(
        seen_seen_df,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    s1_df = s1_df.reset_index(drop=True)
    s2_df = s2_df.reset_index(drop=True)


    train_drugs = (
        set(train_df['Drug1_ID'])
        | set(train_df['Drug2_ID'])
    )
    if not train_drugs.isdisjoint(unseen_drugs):
        raise AssertionError(
            'Unseen drugs have leaked into inductive training data.'
        )

    s1_valid = (
        s1_df['Drug1_ID'].isin(unseen_drugs)
        ^ s1_df['Drug2_ID'].isin(unseen_drugs)
    )
    if not bool(s1_valid.all()):
        raise AssertionError(
            'At least one S1 triplet does not contain exactly one '
            'unseen drug.'
        )

    s2_valid = (
        s2_df['Drug1_ID'].isin(unseen_drugs)
        & s2_df['Drug2_ID'].isin(unseen_drugs)
    )
    if not bool(s2_valid.all()):
        raise AssertionError(
            'At least one S2 triplet does not contain two unseen drugs.'
        )

    return (
        train_df,
        val_df,
        s1_df,
        s2_df,
        seen_drugs,
        unseen_drugs,
    )


def sample_one_negative(
    head,
    tail,
    relation,
    head_pool,
    tail_pool,
    all_positive_triplets,
    used_negative_triplets,
    rng,
    max_trials=10000,
):

    if not head_pool or not tail_pool:
        raise ValueError(
            'Negative-sampling candidate pool is empty.'
        )

    head_pool_array = np.asarray(
        sorted(head_pool),
        dtype=object,
    )
    tail_pool_array = np.asarray(
        sorted(tail_pool),
        dtype=object,
    )

    relation = int(relation)
    head = str(head)
    tail = str(tail)

    for _ in range(max_trials):
        corrupt_head = bool(rng.integers(0, 2))

        if corrupt_head:
            negative_head = str(rng.choice(head_pool_array))
            negative_tail = tail
        else:
            negative_head = head
            negative_tail = str(rng.choice(tail_pool_array))

        candidate = (
            negative_head,
            negative_tail,
            relation,
        )

        # 替换后必须实际发生变化。
        if candidate == (head, tail, relation):
            continue

        # 通常 DDI 数据不包含药物自身交互。
        if negative_head == negative_tail:
            continue

        if candidate in all_positive_triplets:
            continue

        if candidate in used_negative_triplets:
            continue

        used_negative_triplets.add(candidate)
        return candidate

    raise RuntimeError(
        'Unable to generate a unique negative triplet for '
        f'({head}, {tail}, {relation}) after {max_trials} trials.'
    )


def get_candidate_pools(
    head,
    tail,
    scenario,
    all_drugs,
    seen_drugs=None,
    unseen_drugs=None,
):
 
    if scenario == 'transductive':
        return set(all_drugs), set(all_drugs)

    if seen_drugs is None or unseen_drugs is None:
        raise ValueError(
            'seen_drugs and unseen_drugs are required for '
            'inductive negative sampling.'
        )

    if scenario in {'inductive_train', 'inductive_val'}:
        return set(seen_drugs), set(seen_drugs)

    if scenario == 's1':
        head_pool = (
            set(unseen_drugs)
            if head in unseen_drugs
            else set(seen_drugs)
        )
        tail_pool = (
            set(unseen_drugs)
            if tail in unseen_drugs
            else set(seen_drugs)
        )
        return head_pool, tail_pool

    if scenario == 's2':
        return set(unseen_drugs), set(unseen_drugs)

    raise ValueError(f'Unknown scenario: {scenario}')


def build_binary_dataset(
    positive_df,
    scenario,
    all_drugs,
    all_positive_triplets,
    used_negative_triplets,
    rng,
    seen_drugs=None,
    unseen_drugs=None,
):

    rows = []

    for row in tqdm(
        positive_df.itertuples(index=False),
        total=len(positive_df),
        desc=f'Negative sampling: {scenario}',
    ):
        head = str(row.Drug1_ID)
        tail = str(row.Drug2_ID)
        relation = int(row.Y)

        rows.append({
            'Drug1_ID': head,
            'Drug2_ID': tail,
            'Y': relation,
            'label': 1,
        })

        head_pool, tail_pool = get_candidate_pools(
            head=head,
            tail=tail,
            scenario=scenario,
            all_drugs=all_drugs,
            seen_drugs=seen_drugs,
            unseen_drugs=unseen_drugs,
        )

        negative_head, negative_tail, negative_relation = (
            sample_one_negative(
                head=head,
                tail=tail,
                relation=relation,
                head_pool=head_pool,
                tail_pool=tail_pool,
                all_positive_triplets=all_positive_triplets,
                used_negative_triplets=used_negative_triplets,
                rng=rng,
            )
        )

        rows.append({
            'Drug1_ID': negative_head,
            'Drug2_ID': negative_tail,
            'Y': negative_relation,
            'label': 0,
        })

    binary_df = pd.DataFrame(rows)


    permutation = rng.permutation(len(binary_df))
    binary_df = (
        binary_df.iloc[permutation]
        .reset_index(drop=True)
    )

    positive_count = int(binary_df['label'].sum())
    negative_count = int(
        (binary_df['label'] == 0).sum()
    )

    if positive_count != negative_count:
        raise AssertionError(
            f'Positive/negative ratio is not 1:1: '
            f'{positive_count}/{negative_count}'
        )

    return binary_df


def check_split_disjointness(split_dict):

    split_sets = {
        split_name: set(
            zip(
                split_df['Drug1_ID'].astype(str),
                split_df['Drug2_ID'].astype(str),
                split_df['Y'].astype(int),
            )
        )
        for split_name, split_df in split_dict.items()
    }

    names = list(split_sets)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            overlap = (
                split_sets[first_name]
                & split_sets[second_name]
            )
            if overlap:
                raise AssertionError(
                    f'{first_name} and {second_name} contain '
                    f'{len(overlap)} overlapping positive triplets.'
                )


def save_csv(dataframe, filename, args):
    output_file = dataset_dir(args) / filename
    dataframe.to_csv(output_file, index=False)
    print(
        f'Saved {filename}: '
        f'{len(dataframe)} binary instances.'
    )


def write_static_dataset_metadata(
    args,
    positive_df,
    candidate_drugs,
    similarity_dim,
    relation_map,
):
    metadata = {
        'dataset': args.dataset,
        'raw_file': str(
            Path(args.dataset_filename).resolve()
        ),
        'num_positive_triplets': int(len(positive_df)),
        'num_drugs_in_triplets': int(len(candidate_drugs)),
        'similarity_dim': int(similarity_dim),
        'num_relations': int(len(relation_map)),
        'columns': {
            'head_id': args.c_id1,
            'tail_id': args.c_id2,
            'head_smiles': args.c_s1,
            'tail_smiles': args.c_s2,
            'relation': args.c_y,
        },
        'similarity_graph': {
            'morgan_radius': 2,
            'topk': int(args.sim_topk),
            'tau_mode': args.tau_mode,
            'd_min': int(args.d_min),
            'd_max': int(args.d_max),
        },
    }

    save_json(metadata, 'dataset_meta.json', args)


def generate_transductive_splits(
    args,
    positive_df,
    candidate_drugs,
):
    rng = np.random.default_rng(args.seed)

    train_positive, val_positive, test_positive = (
        split_transductive_positive_triplets(
            positive_df=positive_df,
            seed=args.seed,
        )
    )

    positive_splits = {
        'train': train_positive,
        'val': val_positive,
        'test': test_positive,
    }
    check_split_disjointness(positive_splits)

    all_positive_triplets = set(
        zip(
            positive_df['Drug1_ID'].astype(str),
            positive_df['Drug2_ID'].astype(str),
            positive_df['Y'].astype(int),
        )
    )


    used_negative_triplets = set()

    output_counts = {}

    for split_name, split_positive_df in positive_splits.items():
        binary_df = build_binary_dataset(
            positive_df=split_positive_df,
            scenario='transductive',
            all_drugs=candidate_drugs,
            all_positive_triplets=all_positive_triplets,
            used_negative_triplets=used_negative_triplets,
            rng=rng,
        )

        filename = (
            f'transductive_seed{args.seed}_{split_name}.csv'
        )
        save_csv(binary_df, filename, args)

        output_counts[split_name] = {
            'positive': int(len(split_positive_df)),
            'negative': int(len(split_positive_df)),
            'total': int(len(binary_df)),
        }

    split_metadata = {
        'dataset': args.dataset,
        'mode': 'transductive',
        'seed': int(args.seed),
        'positive_split_ratio': {
            'train': 0.60,
            'validation': 0.20,
            'test': 0.20,
        },
        'negative_ratio': 1,
        'counts': output_counts,
    }

    save_json(
        split_metadata,
        f'split_meta_transductive_seed{args.seed}.json',
        args,
    )


def generate_inductive_splits(
    args,
    positive_df,
    candidate_drugs,
):
    rng = np.random.default_rng(args.seed)

    (
        train_positive,
        val_positive,
        s1_positive,
        s2_positive,
        seen_drugs,
        unseen_drugs,
    ) = split_inductive_positive_triplets(
        positive_df=positive_df,
        candidate_drugs=candidate_drugs,
        unseen_ratio=args.unseen_ratio,
        seed=args.seed,
    )

    positive_splits = {
        'train': train_positive,
        'val': val_positive,
        's1': s1_positive,
        's2': s2_positive,
    }
    check_split_disjointness(positive_splits)

    all_positive_triplets = set(
        zip(
            positive_df['Drug1_ID'].astype(str),
            positive_df['Drug2_ID'].astype(str),
            positive_df['Y'].astype(int),
        )
    )
    used_negative_triplets = set()

    scenario_map = {
        'train': 'inductive_train',
        'val': 'inductive_val',
        's1': 's1',
        's2': 's2',
    }

    output_counts = {}

    for split_name, split_positive_df in positive_splits.items():
        binary_df = build_binary_dataset(
            positive_df=split_positive_df,
            scenario=scenario_map[split_name],
            all_drugs=candidate_drugs,
            all_positive_triplets=all_positive_triplets,
            used_negative_triplets=used_negative_triplets,
            rng=rng,
            seen_drugs=seen_drugs,
            unseen_drugs=unseen_drugs,
        )

        filename = (
            f'inductive_seed{args.seed}_{split_name}.csv'
        )
        save_csv(binary_df, filename, args)

        output_counts[split_name] = {
            'positive': int(len(split_positive_df)),
            'negative': int(len(split_positive_df)),
            'total': int(len(binary_df)),
        }

    unseen_file = (
        dataset_dir(args)
        / f'inductive_seed{args.seed}_unseen_drugs.txt'
    )
    unseen_file.write_text(
        '\n'.join(sorted(unseen_drugs)),
        encoding='utf-8',
    )

    split_metadata = {
        'dataset': args.dataset,
        'mode': 'inductive',
        'seed': int(args.seed),
        'unseen_drug_ratio': float(args.unseen_ratio),
        'num_seen_drugs': int(len(seen_drugs)),
        'num_unseen_drugs': int(len(unseen_drugs)),
        'seen_seen_validation_ratio': 0.20,
        'negative_ratio': 1,
        'counts': output_counts,
    }

    save_json(
        split_metadata,
        f'split_meta_inductive_seed{args.seed}.json',
        args,
    )


def generate_splits(args):
    (
        positive_df,
        candidate_drugs,
        similarity_dim,
        relation_map,
    ) = load_valid_positive_triplets(args)

    write_static_dataset_metadata(
        args=args,
        positive_df=positive_df,
        candidate_drugs=candidate_drugs,
        similarity_dim=similarity_dim,
        relation_map=relation_map,
    )

    if args.mode in {'transductive', 'both'}:
        generate_transductive_splits(
            args=args,
            positive_df=positive_df,
            candidate_drugs=candidate_drugs,
        )

    if args.mode in {'inductive', 'both'}:
        generate_inductive_splits(
            args=args,
            positive_df=positive_df,
            candidate_drugs=candidate_drugs,
        )


def decode_delimiter(delimiter):
    if delimiter == r'\t':
        return '\t'
    if delimiter == r'\s':
        return r'\s+'
    return delimiter


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Prepare molecular graphs and paper-consistent '
            'transductive/inductive DDI splits.'
        )
    )

    parser.add_argument(
        '-d',
        '--dataset',
        required=True,
        choices=['drugbank', 'twosides'],
    )
    parser.add_argument(
        '--raw-file',
        required=True,
        help='Path to the raw DrugBank or TwoSIDES table.',
    )
    parser.add_argument(
        '--output-root',
        default='./data/processed',
        help='Parent directory for processed datasets.',
    )
    parser.add_argument(
        '--operation',
        choices=['all', 'drug_data', 'split'],
        default='all',
    )
    parser.add_argument(
        '--mode',
        choices=['transductive', 'inductive', 'both'],
        default='both',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--unseen-ratio',
        type=float,
        default=0.20,
    )


    parser.add_argument('--head-id-col', default=None)
    parser.add_argument('--tail-id-col', default=None)
    parser.add_argument('--head-smiles-col', default=None)
    parser.add_argument('--tail-smiles-col', default=None)
    parser.add_argument('--relation-col', default=None)
    parser.add_argument('--delimiter', default=None)


    parser.add_argument('--sim-topk', type=int, default=32)
    parser.add_argument('--tau-mode', default='p70')
    parser.add_argument('--d-min', type=int, default=8)
    parser.add_argument('--d-max', type=int, default=64)

    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    default_columns = {
        'drugbank': {
            'head_id': 'ID1',
            'tail_id': 'ID2',
            'head_smiles': 'X1',
            'tail_smiles': 'X2',
            'relation': 'Y',
            'delimiter': '\t',
        },
        'twosides': {
            'head_id': 'Drug1_ID',
            'tail_id': 'Drug2_ID',
            'head_smiles': 'Drug1',
            'tail_smiles': 'Drug2',
            'relation': 'New Y',
            'delimiter': ',',
        },
    }

    defaults = default_columns[args.dataset]

    args.c_id1 = args.head_id_col or defaults['head_id']
    args.c_id2 = args.tail_id_col or defaults['tail_id']
    args.c_s1 = (
        args.head_smiles_col or defaults['head_smiles']
    )
    args.c_s2 = (
        args.tail_smiles_col or defaults['tail_smiles']
    )
    args.c_y = args.relation_col or defaults['relation']

    selected_delimiter = (
        args.delimiter
        if args.delimiter is not None
        else defaults['delimiter']
    )
    args.delimiter = decode_delimiter(selected_delimiter)

    args.dataset_filename = args.raw_file
    args.dirname = args.output_root
    args.random_num_gen = np.random.RandomState(args.seed)

    if not 0 < args.unseen_ratio < 1:
        raise ValueError('--unseen-ratio must be between 0 and 1.')

    if args.operation in {'all', 'drug_data'}:
        load_drug_mol_data(
            args,
            topk=args.sim_topk,
            tau_mode=args.tau_mode,
            d_min=args.d_min,
            d_max=args.d_max,
            do_finalize=True,
        )

    if args.operation in {'all', 'split'}:
        generate_splits(args)


