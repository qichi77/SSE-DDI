from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from scipy import sparse as sp
from torch_geometric.data import Data
from tqdm import tqdm


Triplet = Tuple[str, str, int]


class CustomData(Data):
    """PyG data object with correct batching for line-graph edge indices."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "line_graph_edge_index":
            return self.edge_index.size(1) if self.edge_index.numel() else 0
        return super().__inc__(key, value, *args, **kwargs)


def one_of_k_encoding(value: Any, possible_values: Sequence[Any]) -> List[bool]:
    if value not in possible_values:
        raise ValueError(f"{value!r} is not a valid value in {possible_values!r}")
    return [value == candidate for candidate in possible_values]


def one_of_k_encoding_unk(value: Any, allowable_set: Sequence[Any]) -> List[bool]:
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == candidate for candidate in allowable_set]


def atom_features(
    atom: Chem.Atom,
    atom_symbols: Sequence[str],
    explicit_h: bool = True,
    use_chirality: bool = False,
) -> torch.Tensor:
    """Build the atom feature vector used by the public SSE-DDI encoder."""

    features: List[Any] = (
        one_of_k_encoding_unk(atom.GetSymbol(), list(atom_symbols) + ["Unknown"])
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(7)))
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic()]
    )

    if explicit_h:
        features += one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(5)))

    if use_chirality:
        try:
            features += one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
            features += [atom.HasProp("_ChiralityPossible")]
        except (KeyError, RuntimeError):
            features += [False, False, atom.HasProp("_ChiralityPossible")]

    return torch.tensor(np.asarray(features, dtype=np.float32))


def bond_features(bond: Chem.Bond) -> torch.Tensor:
    """Six-dimensional chemical-bond feature vector."""

    bond_type = bond.GetBondType()
    return torch.tensor(
        [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ],
        dtype=torch.float32,
    )


def build_directed_bond_states(molecule: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand every undirected chemical bond into two directed bond states."""

    directed_edges: List[Tuple[int, int]] = []
    directed_features: List[torch.Tensor] = []

    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feature = bond_features(bond)

        directed_edges.append((begin, end))
        directed_features.append(feature)
        directed_edges.append((end, begin))
        directed_features.append(feature.clone())

    if not directed_edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 6), dtype=torch.float32),
        )

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(directed_features, dim=0)
    return edge_index, edge_attr


def build_non_backtracking_line_graph(edge_index: torch.Tensor) -> torch.Tensor:
    """Construct e=(u,v) -> e'=(v,w) only when w != u."""

    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    source_atom = edge_index[0]
    target_atom = edge_index[1]

    consecutive = target_atom.unsqueeze(1) == source_atom.unsqueeze(0)
    immediate_reverse = source_atom.unsqueeze(1) == target_atom.unsqueeze(0)
    connections = consecutive & ~immediate_reverse

    return connections.nonzero(as_tuple=False).t().contiguous()


def build_pyg_graph(
    drug_id: str,
    canonical_smiles: str,
    molecule: Chem.Mol,
    atom_symbols: Sequence[str],
    similarity_row: sp.csr_matrix,
    similarity_dim: int,
) -> CustomData:
    atom_items = [
        (atom.GetIdx(), atom_features(atom, atom_symbols))
        for atom in molecule.GetAtoms()
    ]
    atom_items.sort(key=lambda item: item[0])

    if not atom_items:
        raise ValueError(f"Drug {drug_id!r} contains no atoms")

    x = torch.stack([feature for _, feature in atom_items], dim=0)
    edge_index, edge_attr = build_directed_bond_states(molecule)
    line_graph_edge_index = build_non_backtracking_line_graph(edge_index)

    similarity = torch.zeros(similarity_dim, dtype=torch.float32)
    if similarity_row.nnz:
        similarity[torch.from_numpy(similarity_row.indices).long()] = torch.from_numpy(
            similarity_row.data.astype(np.float32, copy=False)
        )

    return CustomData(
        x=x,
        edge_index=edge_index,
        line_graph_edge_index=line_graph_edge_index,
        edge_attr=edge_attr,
        sim=similarity.unsqueeze(0),
        id=str(drug_id),
        smiles=canonical_smiles,
    )


def build_dgl_graph(molecule: Chem.Mol, atom_symbols: Sequence[str]) -> dgl.DGLGraph:
    atom_items = [
        (atom.GetIdx(), atom_features(atom, atom_symbols))
        for atom in molecule.GetAtoms()
    ]
    atom_items.sort(key=lambda item: item[0])

    if not atom_items:
        raise ValueError("Molecule contains no atoms")

    node_features = torch.stack([feature for _, feature in atom_items], dim=0)
    edge_index, edge_attr = build_directed_bond_states(molecule)

    if edge_index.numel():
        source, target = edge_index[0], edge_index[1]
    else:
        source = torch.empty(0, dtype=torch.long)
        target = torch.empty(0, dtype=torch.long)

    graph = dgl.graph((source, target), num_nodes=node_features.size(0))
    graph.ndata["feat"] = node_features
    graph.edata["feat"] = edge_attr
    return graph


def decode_delimiter(delimiter: str) -> str:
    if delimiter == r"\t":
        return "\t"
    if delimiter == r"\s":
        return r"\s+"
    return delimiter


def read_raw_table(filename: str, delimiter: str) -> pd.DataFrame:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(path)

    kwargs: Dict[str, Any] = {
        "sep": delimiter,
        "dtype": str,
        "keep_default_na": False,
    }
    if delimiter == r"\s+":
        kwargs["engine"] = "python"

    return pd.read_csv(path, **kwargs)


def require_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Existing columns: {list(dataframe.columns)}"
        )


def normalize_identifier(value: Any) -> str:
    return str(value).strip()


def canonicalize_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    text = str(smiles).strip()
    if not text:
        return None, None

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None, None

    canonical = Chem.MolToSmiles(molecule, canonical=True)
    canonical_molecule = Chem.MolFromSmiles(canonical)
    if canonical_molecule is None:
        return None, None

    return canonical_molecule, canonical


def collect_benchmark_molecules(
    dataframe: pd.DataFrame,
    head_id_col: str,
    tail_id_col: str,
    head_smiles_col: str,
    tail_smiles_col: str,
    relation_col: str,
    allow_smiles_conflicts: bool,
) -> Tuple[
    List[str],
    Dict[str, Chem.Mol],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    int,
]:
    """Collect valid molecules and retain drugs occurring in a valid DDI row."""

    require_columns(
        dataframe,
        [head_id_col, tail_id_col, head_smiles_col, tail_smiles_col, relation_col],
    )

    smiles_candidates: Dict[str, Set[str]] = {}
    for id_col, smiles_col in (
        (head_id_col, head_smiles_col),
        (tail_id_col, tail_smiles_col),
    ):
        for drug_id_raw, smiles_raw in zip(dataframe[id_col], dataframe[smiles_col]):
            drug_id = normalize_identifier(drug_id_raw)
            smiles = str(smiles_raw).strip()
            if not drug_id:
                continue
            smiles_candidates.setdefault(drug_id, set()).add(smiles)

    molecule_by_id: Dict[str, Chem.Mol] = {}
    canonical_smiles_by_id: Dict[str, str] = {}
    invalid_smiles_by_id: Dict[str, List[str]] = {}
    conflicting_smiles_by_id: Dict[str, List[str]] = {}

    for drug_id in sorted(smiles_candidates):
        valid_canonical: Dict[str, Chem.Mol] = {}
        invalid_raw: List[str] = []

        for raw_smiles in sorted(smiles_candidates[drug_id]):
            molecule, canonical = canonicalize_smiles(raw_smiles)
            if molecule is None or canonical is None:
                invalid_raw.append(raw_smiles)
            else:
                valid_canonical.setdefault(canonical, molecule)

        if invalid_raw:
            invalid_smiles_by_id[drug_id] = invalid_raw

        if not valid_canonical:
            continue

        canonical_values = sorted(valid_canonical)
        if len(canonical_values) > 1:
            conflicting_smiles_by_id[drug_id] = canonical_values
            if not allow_smiles_conflicts:
                raise ValueError(
                    f"Drug {drug_id!r} has multiple non-equivalent SMILES: "
                    f"{canonical_values}. Resolve the raw data or pass "
                    "--allow-smiles-conflicts to select the lexicographically first "
                    "canonical SMILES deterministically."
                )

        selected = canonical_values[0]
        molecule_by_id[drug_id] = valid_canonical[selected]
        canonical_smiles_by_id[drug_id] = selected

    # The benchmark dictionary is restricted to drugs appearing in at least one
    # observed triplet whose two drugs have valid molecular structures.
    benchmark_drugs: Set[str] = set()
    valid_rows = 0

    for head_raw, tail_raw, relation_raw in zip(
        dataframe[head_id_col], dataframe[tail_id_col], dataframe[relation_col]
    ):
        head = normalize_identifier(head_raw)
        tail = normalize_identifier(tail_raw)
        relation = str(relation_raw).strip()

        if not head or not tail or not relation:
            continue
        if head not in molecule_by_id or tail not in molecule_by_id:
            continue

        benchmark_drugs.add(head)
        benchmark_drugs.add(tail)
        valid_rows += 1

    if not benchmark_drugs:
        raise ValueError("No valid DDI rows remain after SMILES validation")

    drug_ids = sorted(benchmark_drugs)
    molecule_by_id = {drug_id: molecule_by_id[drug_id] for drug_id in drug_ids}
    canonical_smiles_by_id = {
        drug_id: canonical_smiles_by_id[drug_id] for drug_id in drug_ids
    }

    return (
        drug_ids,
        molecule_by_id,
        canonical_smiles_by_id,
        invalid_smiles_by_id,
        conflicting_smiles_by_id,
        valid_rows,
    )


def make_morgan_fingerprint(
    molecule: Chem.Mol,
    radius: int,
    n_bits: int,
) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius,
        nBits=n_bits,
    )


def stable_descending_indices(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sort by score descending and index ascending for deterministic ties."""

    return np.lexsort((indices, -values))


def refine_local_similarity_row(
    similarities: np.ndarray,
    self_index: int,
    topk: int,
    quantile: float,
    std_lambda: float,
    d_min: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    row = np.asarray(similarities, dtype=np.float32).copy()
    row[self_index] = 0.0

    positive_indices = np.flatnonzero(row > 0.0)
    if positive_indices.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            {"tau_q": 0.0, "tau_mu": 0.0, "tau": 0.0},
        )

    positive_values = row[positive_indices]
    tau_q = float(np.quantile(positive_values, quantile))
    tau_mu = float(positive_values.mean() + std_lambda * positive_values.std(ddof=0))
    tau = max(tau_q, tau_mu)

    order = stable_descending_indices(positive_values, positive_indices)
    candidate_indices = positive_indices[order][:topk]
    candidate_values = row[candidate_indices]

    threshold_mask = candidate_values >= tau
    retained_indices = candidate_indices[threshold_mask].tolist()
    retained_values = candidate_values[threshold_mask].tolist()

    if len(retained_indices) < d_min:
        for index, value in zip(
            candidate_indices[~threshold_mask], candidate_values[~threshold_mask]
        ):
            if value <= 0.0:
                continue
            retained_indices.append(int(index))
            retained_values.append(float(value))
            if len(retained_indices) >= d_min:
                break

    return (
        np.asarray(retained_indices, dtype=np.int64),
        np.asarray(retained_values, dtype=np.float32),
        {"tau_q": tau_q, "tau_mu": tau_mu, "tau": tau},
    )


def build_local_sparse_similarity(
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
    topk: int,
    quantile: float,
    std_lambda: float,
    d_min: int,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    num_drugs = len(fingerprints)
    matrix = sp.lil_matrix((num_drugs, num_drugs), dtype=np.float32)

    tau_q_values: List[float] = []
    tau_mu_values: List[float] = []
    tau_values: List[float] = []

    for row_index, fingerprint in enumerate(
        tqdm(fingerprints, desc="Computing Morgan/Tanimoto similarity")
    ):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints),
            dtype=np.float32,
        )
        indices, values, thresholds = refine_local_similarity_row(
            similarities=similarities,
            self_index=row_index,
            topk=topk,
            quantile=quantile,
            std_lambda=std_lambda,
            d_min=d_min,
        )

        matrix.rows[row_index] = indices.tolist()
        matrix.data[row_index] = values.tolist()

        tau_q_values.append(thresholds["tau_q"])
        tau_mu_values.append(thresholds["tau_mu"])
        tau_values.append(thresholds["tau"])

    csr = matrix.tocsr()
    csr.eliminate_zeros()
    csr.sort_indices()

    row_degrees = np.diff(csr.indptr)
    threshold_metadata = {
        "tau_q_mean": float(np.mean(tau_q_values)) if tau_q_values else 0.0,
        "tau_mu_mean": float(np.mean(tau_mu_values)) if tau_mu_values else 0.0,
        "tau_mean": float(np.mean(tau_values)) if tau_values else 0.0,
        "tau_min": float(np.min(tau_values)) if tau_values else 0.0,
        "tau_max": float(np.max(tau_values)) if tau_values else 0.0,
        "rows_below_d_min_after_local_backfill": int(np.sum(row_degrees < d_min)),
    }
    return csr, threshold_metadata


def mutual_knn_with_symmetric_degree_cap(
    local_sparse: sp.csr_matrix,
    d_max: int,
) -> sp.csr_matrix:
    """Keep mutual edges and greedily enforce an undirected degree cap."""

    mutual = local_sparse.minimum(local_sparse.transpose()).tocoo()
    mutual.eliminate_zeros()

    candidates: List[Tuple[float, int, int]] = []
    for row, col, value in zip(mutual.row, mutual.col, mutual.data):
        row_i = int(row)
        col_i = int(col)
        if row_i < col_i and value > 0.0:
            candidates.append((float(value), row_i, col_i))

    # Highest-weight mutual edges are retained first. The tie break is stable.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    num_nodes = local_sparse.shape[0]
    degrees = np.zeros(num_nodes, dtype=np.int64)
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    for value, node_i, node_j in candidates:
        if degrees[node_i] >= d_max or degrees[node_j] >= d_max:
            continue

        rows.extend([node_i, node_j])
        cols.extend([node_j, node_i])
        values.extend([value, value])
        degrees[node_i] += 1
        degrees[node_j] += 1

    capped = sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=local_sparse.shape,
        dtype=np.float32,
    )
    capped.eliminate_zeros()
    capped.sort_indices()
    return capped


def second_degree_backfill(
    capped_mutual: sp.csr_matrix,
    local_sparse: sp.csr_matrix,
    d_min: int,
) -> sp.csr_matrix:
    """Backfill low-degree rows from their pre-mutual sparse candidate rows."""

    output = capped_mutual.tolil(copy=True)
    local = local_sparse.tocsr()

    for row_index in range(output.shape[0]):
        if len(output.rows[row_index]) >= d_min:
            continue

        present = set(output.rows[row_index])
        source_row = local.getrow(row_index)
        if source_row.nnz == 0:
            continue

        order = stable_descending_indices(source_row.data, source_row.indices)
        for position in order:
            column = int(source_row.indices[position])
            value = float(source_row.data[position])

            if column == row_index or column in present or value <= 0.0:
                continue

            output.rows[row_index].append(column)
            output.data[row_index].append(value)
            present.add(column)

            if len(output.rows[row_index]) >= d_min:
                break

    result = output.tocsr()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def refine_similarity_graph(
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
    topk: int,
    quantile: float,
    std_lambda: float,
    d_min: int,
    d_max: int,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    local_sparse, threshold_metadata = build_local_sparse_similarity(
        fingerprints=fingerprints,
        topk=topk,
        quantile=quantile,
        std_lambda=std_lambda,
        d_min=d_min,
    )

    capped_mutual = mutual_knn_with_symmetric_degree_cap(local_sparse, d_max=d_max)
    refined = second_degree_backfill(
        capped_mutual=capped_mutual,
        local_sparse=local_sparse,
        d_min=d_min,
    )

    local_degrees = np.diff(local_sparse.indptr)
    mutual_degrees = np.diff(capped_mutual.indptr)
    final_degrees = np.diff(refined.indptr)

    metadata: Dict[str, Any] = {
        **threshold_metadata,
        "local_nonzero_entries": int(local_sparse.nnz),
        "mutual_capped_nonzero_entries": int(capped_mutual.nnz),
        "final_nonzero_entries": int(refined.nnz),
        "local_mean_out_degree": float(local_degrees.mean()) if local_degrees.size else 0.0,
        "mutual_capped_mean_degree": (
            float(mutual_degrees.mean()) if mutual_degrees.size else 0.0
        ),
        "final_mean_out_degree": float(final_degrees.mean()) if final_degrees.size else 0.0,
        "final_max_out_degree": int(final_degrees.max()) if final_degrees.size else 0,
        "rows_below_d_min_after_global_backfill": int(np.sum(final_degrees < d_min)),
    }
    return refined, metadata


def molecular_graph_statistics(molecules: Sequence[Chem.Mol]) -> Dict[str, Any]:
    atom_counts: List[int] = []
    bond_counts: List[int] = []
    line_edge_counts: List[int] = []
    expansion_ratios: List[float] = []
    maximum_degrees: List[int] = []

    for molecule in molecules:
        num_atoms = molecule.GetNumAtoms()
        num_bonds = molecule.GetNumBonds()
        edge_index, _ = build_directed_bond_states(molecule)
        line_graph_edge_index = build_non_backtracking_line_graph(edge_index)
        num_line_edges = line_graph_edge_index.size(1)
        ratio = num_line_edges / (2.0 * num_bonds) if num_bonds > 0 else 0.0
        max_degree = max((atom.GetDegree() for atom in molecule.GetAtoms()), default=0)

        atom_counts.append(num_atoms)
        bond_counts.append(num_bonds)
        line_edge_counts.append(num_line_edges)
        expansion_ratios.append(ratio)
        maximum_degrees.append(max_degree)

    ratios = np.asarray(expansion_ratios, dtype=np.float64)
    return {
        "avg_num_atoms": float(np.mean(atom_counts)),
        "avg_num_undirected_bonds": float(np.mean(bond_counts)),
        "avg_num_directed_line_graph_edges": float(np.mean(line_edge_counts)),
        "line_graph_expansion_ratio_mean": float(np.mean(ratios)),
        "line_graph_expansion_ratio_median": float(np.median(ratios)),
        "line_graph_expansion_ratio_p95": float(np.quantile(ratios, 0.95)),
        "line_graph_expansion_ratio_max": float(np.max(ratios)),
        "maximum_atom_degree": int(np.max(maximum_degrees)),
    }


def dataset_directory(args: argparse.Namespace) -> Path:
    path = Path(args.output_root) / args.dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(data: Any, filename: str, args: argparse.Namespace) -> None:
    output_file = dataset_directory(args) / filename
    with output_file.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output_file}")


def save_json(data: Any, filename: str, args: argparse.Namespace) -> None:
    output_file = dataset_directory(args) / filename
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Saved {output_file}")


def save_csv(dataframe: pd.DataFrame, filename: str, args: argparse.Namespace) -> None:
    output_file = dataset_directory(args) / filename
    dataframe.to_csv(output_file, index=False)
    print(f"Saved {output_file}: {len(dataframe)} instances")


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(filename: str) -> str:
    digest = hashlib.sha256()
    with Path(filename).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_drug_graphs(args: argparse.Namespace) -> None:
    dataframe = read_raw_table(args.raw_file, args.delimiter)

    (
        drug_ids,
        molecule_by_id,
        canonical_smiles_by_id,
        invalid_smiles_by_id,
        conflicting_smiles_by_id,
        valid_raw_rows,
    ) = collect_benchmark_molecules(
        dataframe=dataframe,
        head_id_col=args.head_id_col,
        tail_id_col=args.tail_id_col,
        head_smiles_col=args.head_smiles_col,
        tail_smiles_col=args.tail_smiles_col,
        relation_col=args.relation_col,
        allow_smiles_conflicts=args.allow_smiles_conflicts,
    )

    molecules = [molecule_by_id[drug_id] for drug_id in drug_ids]
    atom_symbols = sorted(
        {atom.GetSymbol() for molecule in molecules for atom in molecule.GetAtoms()}
    )

    fingerprints = [
        make_morgan_fingerprint(
            molecule=molecule,
            radius=args.morgan_radius,
            n_bits=args.morgan_nbits,
        )
        for molecule in molecules
    ]

    refined_similarity, similarity_metadata = refine_similarity_graph(
        fingerprints=fingerprints,
        topk=args.sim_topk,
        quantile=args.sim_quantile,
        std_lambda=args.sim_std_lambda,
        d_min=args.d_min,
        d_max=args.d_max,
    )

    drug_data_pyg: Dict[str, CustomData] = {}
    drug_data_dgl: Dict[str, dgl.DGLGraph] = {}

    for index, drug_id in enumerate(tqdm(drug_ids, desc="Building molecular graphs")):
        molecule = molecule_by_id[drug_id]
        drug_data_pyg[drug_id] = build_pyg_graph(
            drug_id=drug_id,
            canonical_smiles=canonical_smiles_by_id[drug_id],
            molecule=molecule,
            atom_symbols=atom_symbols,
            similarity_row=refined_similarity.getrow(index),
            similarity_dim=len(drug_ids),
        )
        drug_data_dgl[drug_id] = build_dgl_graph(molecule, atom_symbols)

    save_pickle(drug_data_pyg, "drug_data_pyg.pkl", args)
    save_pickle(drug_data_dgl, "drug_data_dgl.pkl", args)
    save_json({drug_id: index for index, drug_id in enumerate(drug_ids)}, "drug_index.json", args)
    save_json(atom_symbols, "atom_symbols.json", args)
    save_json(invalid_smiles_by_id, "invalid_smiles.json", args)
    save_json(conflicting_smiles_by_id, "smiles_conflicts.json", args)

    graph_metadata: Dict[str, Any] = {
        "dataset": args.dataset,
        "raw_file": str(Path(args.raw_file).resolve()),
        "raw_file_sha256": sha256_file(args.raw_file),
        "raw_row_count": int(len(dataframe)),
        "raw_rows_with_two_valid_molecules_and_relation": int(valid_raw_rows),
        "num_drugs": int(len(drug_ids)),
        "similarity_dim": int(len(drug_ids)),
        "num_atom_symbols": int(len(atom_symbols)),
        "atom_symbols": atom_symbols,
        "invalid_smiles_drug_count": int(len(invalid_smiles_by_id)),
        "conflicting_smiles_drug_count": int(len(conflicting_smiles_by_id)),
        "rdkit_version": rdBase.rdkitVersion,
        "columns": {
            "head_id": args.head_id_col,
            "tail_id": args.tail_id_col,
            "head_smiles": args.head_smiles_col,
            "tail_smiles": args.tail_smiles_col,
            "relation": args.relation_col,
        },
        "morgan_fingerprint": {
            "radius": int(args.morgan_radius),
            "n_bits": int(args.morgan_nbits),
        },
        "similarity_refinement": {
            "topk": int(args.sim_topk),
            "threshold_formula": "max(tau_q, tau_mu)",
            "tau_q": "quantile of positive non-self similarities",
            "quantile": float(args.sim_quantile),
            "tau_mu": "mean + lambda * population_std",
            "std_lambda": float(args.sim_std_lambda),
            "d_min": int(args.d_min),
            "d_max": int(args.d_max),
            "mutual_edge_weight": "minimum of the two directed similarities",
            "degree_cap": "greedy symmetric highest-weight selection",
            **similarity_metadata,
        },
        "molecular_graph_statistics": molecular_graph_statistics(molecules),
    }
    save_json(graph_metadata, "graph_meta.json", args)
    save_json(graph_metadata, "dataset_meta.json", args)


def relation_sort_key(value: str) -> Tuple[int, Any]:
    text = str(value).strip()
    try:
        return 0, Decimal(text)
    except InvalidOperation:
        return 1, text


def load_valid_positive_triplets(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str], int, Dict[str, int]]:
    output_dir = dataset_directory(args)
    graph_file = output_dir / "drug_data_pyg.pkl"
    index_file = output_dir / "drug_index.json"

    if not graph_file.exists() or not index_file.exists():
        raise FileNotFoundError(
            "Molecular graph files do not exist. Run data_pre.py with "
            "--operation drug_data first."
        )

    with graph_file.open("rb") as file:
        drug_graph = pickle.load(file)
    drug_index = load_json_file(index_file)

    valid_drug_ids = {normalize_identifier(drug_id) for drug_id in drug_graph}
    if valid_drug_ids != set(drug_index):
        raise ValueError("drug_data_pyg.pkl and drug_index.json contain different drug IDs")

    dataframe = read_raw_table(args.raw_file, args.delimiter)
    require_columns(
        dataframe,
        [args.head_id_col, args.tail_id_col, args.relation_col],
    )

    working = dataframe[
        [args.head_id_col, args.tail_id_col, args.relation_col]
    ].copy()
    working[args.head_id_col] = working[args.head_id_col].map(normalize_identifier)
    working[args.tail_id_col] = working[args.tail_id_col].map(normalize_identifier)
    working[args.relation_col] = working[args.relation_col].astype(str).str.strip()

    valid_mask = (
        working[args.head_id_col].isin(valid_drug_ids)
        & working[args.tail_id_col].isin(valid_drug_ids)
        & working[args.relation_col].ne("")
    )
    working = working.loc[valid_mask].copy()

    if working.empty:
        raise ValueError("No valid positive triplets remain after molecular filtering")

    relation_values = sorted(
        working[args.relation_col].unique().tolist(),
        key=relation_sort_key,
    )
    relation_map = {
        str(raw_relation): relation_id
        for relation_id, raw_relation in enumerate(relation_values)
    }

    positive_df = pd.DataFrame(
        {
            "Drug1_ID": working[args.head_id_col].astype(str),
            "Drug2_ID": working[args.tail_id_col].astype(str),
            "Y": working[args.relation_col].map(relation_map).astype(np.int64),
        }
    )
    positive_df = positive_df.drop_duplicates(
        subset=["Drug1_ID", "Drug2_ID", "Y"], keep="first"
    ).reset_index(drop=True)

    candidate_drugs = sorted(
        set(positive_df["Drug1_ID"]) | set(positive_df["Drug2_ID"])
    )

    # All prepared graph drugs should occur in at least one retained positive row.
    if set(candidate_drugs) != valid_drug_ids:
        missing_from_triplets = sorted(valid_drug_ids - set(candidate_drugs))
        raise ValueError(
            "Prepared drug dictionary contains drugs absent from retained positives: "
            f"{missing_from_triplets[:10]}"
        )

    save_json(
        {
            "raw_to_id": relation_map,
            "id_to_raw": relation_values,
            "num_relations": len(relation_values),
        },
        "relation_map.json",
        args,
    )

    return positive_df, candidate_drugs, len(drug_index), relation_map


def allocate_counts(total: int, ratios: Sequence[float]) -> List[int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("ratios must sum to 1")

    raw = np.asarray(ratios, dtype=np.float64) * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())

    fractional = raw - counts
    order = np.lexsort((np.arange(len(ratios)), -fractional))
    for index in order[:remainder]:
        counts[index] += 1

    return counts.astype(int).tolist()


def split_dataframe_by_ratios(
    dataframe: pd.DataFrame,
    ratios: Sequence[float],
    seed: int,
) -> List[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(dataframe))
    counts = allocate_counts(len(dataframe), ratios)

    outputs: List[pd.DataFrame] = []
    start = 0
    for count in counts:
        selected = permutation[start : start + count]
        outputs.append(dataframe.iloc[selected].reset_index(drop=True))
        start += count

    return outputs


def split_transductive_positive_triplets(
    positive_df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = split_dataframe_by_ratios(
        positive_df,
        ratios=(0.60, 0.20, 0.20),
        seed=seed,
    )
    return train_df, val_df, test_df


def choose_inductive_partition(
    positive_df: pd.DataFrame,
    candidate_drugs: Sequence[str],
    unseen_ratio: float,
    validation_ratio: float,
    seed: int,
    max_attempts: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Set[str],
    Set[str],
    int,
]:
    num_drugs = len(candidate_drugs)
    num_unseen = int(math.floor(num_drugs * unseen_ratio + 0.5))
    num_unseen = min(max(num_unseen, 1), num_drugs - 1)

    rng = np.random.default_rng(seed)
    drug_array = np.asarray(sorted(candidate_drugs), dtype=object)

    for attempt in range(1, max_attempts + 1):
        unseen_drugs = set(
            rng.choice(drug_array, size=num_unseen, replace=False).tolist()
        )
        seen_drugs = set(candidate_drugs) - unseen_drugs

        head_unseen = positive_df["Drug1_ID"].isin(unseen_drugs)
        tail_unseen = positive_df["Drug2_ID"].isin(unseen_drugs)

        seen_seen = positive_df.loc[~head_unseen & ~tail_unseen].copy()
        s1 = positive_df.loc[head_unseen ^ tail_unseen].copy()
        s2 = positive_df.loc[head_unseen & tail_unseen].copy()

        if len(seen_seen) < 2 or s1.empty or s2.empty:
            continue

        train, validation = split_dataframe_by_ratios(
            seen_seen,
            ratios=(1.0 - validation_ratio, validation_ratio),
            seed=seed,
        )
        if train.empty or validation.empty:
            continue

        return (
            train,
            validation,
            s1.reset_index(drop=True),
            s2.reset_index(drop=True),
            seen_drugs,
            unseen_drugs,
            attempt,
        )

    raise RuntimeError(
        "Unable to construct a non-empty inductive train/validation/S1/S2 split "
        f"after {max_attempts} attempts."
    )


def triplet_set(dataframe: pd.DataFrame) -> Set[Triplet]:
    return set(
        zip(
            dataframe["Drug1_ID"].astype(str),
            dataframe["Drug2_ID"].astype(str),
            dataframe["Y"].astype(int),
        )
    )


def check_positive_split_disjointness(splits: Mapping[str, pd.DataFrame]) -> None:
    split_sets = {name: triplet_set(dataframe) for name, dataframe in splits.items()}
    names = list(split_sets)

    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            overlap = split_sets[first_name] & split_sets[second_name]
            if overlap:
                raise AssertionError(
                    f"{first_name} and {second_name} contain "
                    f"{len(overlap)} overlapping positive triplets"
                )


def candidate_pools_for_scenario(
    head: str,
    tail: str,
    scenario: str,
    all_drugs: Set[str],
    seen_drugs: Optional[Set[str]],
    unseen_drugs: Optional[Set[str]],
) -> Tuple[Set[str], Set[str]]:
    if scenario == "transductive":
        return all_drugs, all_drugs

    if seen_drugs is None or unseen_drugs is None:
        raise ValueError("seen_drugs and unseen_drugs are required for inductive sampling")

    if scenario in {"inductive_train", "inductive_val"}:
        return seen_drugs, seen_drugs

    if scenario == "s1":
        head_pool = unseen_drugs if head in unseen_drugs else seen_drugs
        tail_pool = unseen_drugs if tail in unseen_drugs else seen_drugs
        return head_pool, tail_pool

    if scenario == "s2":
        return unseen_drugs, unseen_drugs

    raise ValueError(f"Unknown negative-sampling scenario: {scenario}")


def valid_negative_candidate(
    candidate: Triplet,
    original: Triplet,
    all_positive_triplets: Set[Triplet],
    used_negative_triplets: Set[Triplet],
) -> bool:
    head, tail, _ = candidate
    return (
        candidate != original
        and head != tail
        and candidate not in all_positive_triplets
        and candidate not in used_negative_triplets
    )


def sample_one_negative(
    head: str,
    tail: str,
    relation: int,
    head_pool: Set[str],
    tail_pool: Set[str],
    all_positive_triplets: Set[Triplet],
    used_negative_triplets: Set[Triplet],
    rng: np.random.Generator,
    max_random_trials: int,
) -> Triplet:
    original = (head, tail, relation)
    head_candidates = np.asarray(sorted(head_pool), dtype=object)
    tail_candidates = np.asarray(sorted(tail_pool), dtype=object)

    if head_candidates.size == 0 or tail_candidates.size == 0:
        raise ValueError("Negative-sampling candidate pool is empty")

    for _ in range(max_random_trials):
        if bool(rng.integers(0, 2)):
            candidate = (str(rng.choice(head_candidates)), tail, relation)
        else:
            candidate = (head, str(rng.choice(tail_candidates)), relation)

        if valid_negative_candidate(
            candidate,
            original,
            all_positive_triplets,
            used_negative_triplets,
        ):
            used_negative_triplets.add(candidate)
            return candidate

    # Deterministic finite fallback prevents a false failure in dense relations.
    fallback: List[Triplet] = [
        (str(candidate_head), tail, relation) for candidate_head in head_candidates
    ] + [(head, str(candidate_tail), relation) for candidate_tail in tail_candidates]

    for position in rng.permutation(len(fallback)):
        candidate = fallback[int(position)]
        if valid_negative_candidate(
            candidate,
            original,
            all_positive_triplets,
            used_negative_triplets,
        ):
            used_negative_triplets.add(candidate)
            return candidate

    raise RuntimeError(
        "Unable to generate a unique negative for "
        f"({head}, {tail}, {relation}) under the current candidate pools"
    )


def build_binary_dataset(
    positive_df: pd.DataFrame,
    scenario: str,
    all_drugs: Set[str],
    all_positive_triplets: Set[Triplet],
    used_negative_triplets: Set[Triplet],
    rng: np.random.Generator,
    max_random_trials: int,
    seen_drugs: Optional[Set[str]] = None,
    unseen_drugs: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for row in tqdm(
        positive_df.itertuples(index=False),
        total=len(positive_df),
        desc=f"Negative sampling: {scenario}",
    ):
        head = str(row.Drug1_ID)
        tail = str(row.Drug2_ID)
        relation = int(row.Y)

        rows.append(
            {"Drug1_ID": head, "Drug2_ID": tail, "Y": relation, "label": 1}
        )

        head_pool, tail_pool = candidate_pools_for_scenario(
            head=head,
            tail=tail,
            scenario=scenario,
            all_drugs=all_drugs,
            seen_drugs=seen_drugs,
            unseen_drugs=unseen_drugs,
        )
        negative_head, negative_tail, negative_relation = sample_one_negative(
            head=head,
            tail=tail,
            relation=relation,
            head_pool=head_pool,
            tail_pool=tail_pool,
            all_positive_triplets=all_positive_triplets,
            used_negative_triplets=used_negative_triplets,
            rng=rng,
            max_random_trials=max_random_trials,
        )
        rows.append(
            {
                "Drug1_ID": negative_head,
                "Drug2_ID": negative_tail,
                "Y": negative_relation,
                "label": 0,
            }
        )

    binary = pd.DataFrame(rows)
    binary = binary.iloc[rng.permutation(len(binary))].reset_index(drop=True)

    positives = int((binary["label"] == 1).sum())
    negatives = int((binary["label"] == 0).sum())
    if positives != negatives:
        raise AssertionError(f"Positive/negative ratio is not 1:1: {positives}/{negatives}")

    negative_rows = binary.loc[binary["label"] == 0]
    if negative_rows.duplicated(subset=["Drug1_ID", "Drug2_ID", "Y"]).any():
        raise AssertionError("Duplicate negative triplets remain after sampling")

    if triplet_set(negative_rows) & all_positive_triplets:
        raise AssertionError("A generated negative occurs in the complete positive set")

    return binary


def update_dataset_metadata(
    args: argparse.Namespace,
    positive_df: pd.DataFrame,
    similarity_dim: int,
    relation_map: Mapping[str, int],
) -> None:
    output_dir = dataset_directory(args)
    metadata = load_json_file(output_dir / "graph_meta.json")
    metadata.update(
        {
            "num_positive_triplets": int(len(positive_df)),
            "num_relations": int(len(relation_map)),
            "similarity_dim": int(similarity_dim),
            "positive_triplet_definition": "ordered (head, tail, relation) tuple",
            "negative_sampling": {
                "positive_to_negative_ratio": "1:1",
                "corruption": "uniform random replacement of head or tail",
                "relation_is_unchanged": True,
                "reject_complete_observed_positive_set": True,
                "reject_duplicate_negatives": True,
                "reject_self_pairs": True,
            },
        }
    )
    save_json(metadata, "dataset_meta.json", args)


def generate_transductive_splits(
    args: argparse.Namespace,
    positive_df: pd.DataFrame,
    candidate_drugs: Sequence[str],
) -> None:
    train_positive, val_positive, test_positive = split_transductive_positive_triplets(
        positive_df=positive_df,
        seed=args.seed,
    )
    positive_splits = {
        "train": train_positive,
        "val": val_positive,
        "test": test_positive,
    }
    check_positive_split_disjointness(positive_splits)

    all_positive_triplets = triplet_set(positive_df)
    used_negative_triplets: Set[Triplet] = set()
    all_drugs = set(candidate_drugs)
    rng = np.random.default_rng(args.seed)
    counts: Dict[str, Any] = {}

    for split_name, split_positive in positive_splits.items():
        binary = build_binary_dataset(
            positive_df=split_positive,
            scenario="transductive",
            all_drugs=all_drugs,
            all_positive_triplets=all_positive_triplets,
            used_negative_triplets=used_negative_triplets,
            rng=rng,
            max_random_trials=args.max_negative_trials,
        )
        save_csv(binary, f"transductive_seed{args.seed}_{split_name}.csv", args)
        counts[split_name] = {
            "positive": int(len(split_positive)),
            "negative": int(len(split_positive)),
            "total": int(len(binary)),
            "positive_fraction_of_all_positives": float(
                len(split_positive) / len(positive_df)
            ),
        }

    if len(used_negative_triplets) != len(positive_df):
        raise AssertionError("Negative triplets are not globally unique across splits")

    save_json(
        {
            "dataset": args.dataset,
            "mode": "transductive",
            "seed": int(args.seed),
            "requested_positive_split_ratio": {
                "train": 0.60,
                "validation": 0.20,
                "test": 0.20,
            },
            "integer_allocation": "largest remainder",
            "negative_ratio": 1,
            "counts": counts,
        },
        f"split_meta_transductive_seed{args.seed}.json",
        args,
    )


def generate_inductive_splits(
    args: argparse.Namespace,
    positive_df: pd.DataFrame,
    candidate_drugs: Sequence[str],
) -> None:
    (
        train_positive,
        val_positive,
        s1_positive,
        s2_positive,
        seen_drugs,
        unseen_drugs,
        split_attempt,
    ) = choose_inductive_partition(
        positive_df=positive_df,
        candidate_drugs=candidate_drugs,
        unseen_ratio=args.unseen_ratio,
        validation_ratio=args.inductive_val_ratio,
        seed=args.seed,
        max_attempts=args.max_inductive_split_attempts,
    )

    positive_splits = {
        "train": train_positive,
        "val": val_positive,
        "s1": s1_positive,
        "s2": s2_positive,
    }
    check_positive_split_disjointness(positive_splits)

    all_positive_triplets = triplet_set(positive_df)
    used_negative_triplets: Set[Triplet] = set()
    all_drugs = set(candidate_drugs)
    rng = np.random.default_rng(args.seed)
    scenario_by_split = {
        "train": "inductive_train",
        "val": "inductive_val",
        "s1": "s1",
        "s2": "s2",
    }
    counts: Dict[str, Any] = {}

    for split_name, split_positive in positive_splits.items():
        binary = build_binary_dataset(
            positive_df=split_positive,
            scenario=scenario_by_split[split_name],
            all_drugs=all_drugs,
            all_positive_triplets=all_positive_triplets,
            used_negative_triplets=used_negative_triplets,
            rng=rng,
            max_random_trials=args.max_negative_trials,
            seen_drugs=seen_drugs,
            unseen_drugs=unseen_drugs,
        )
        save_csv(binary, f"inductive_seed{args.seed}_{split_name}.csv", args)
        counts[split_name] = {
            "positive": int(len(split_positive)),
            "negative": int(len(split_positive)),
            "total": int(len(binary)),
        }

    if len(used_negative_triplets) != sum(len(frame) for frame in positive_splits.values()):
        raise AssertionError("Negative triplets are not globally unique across splits")

    unseen_file = dataset_directory(args) / f"inductive_seed{args.seed}_unseen_drugs.txt"
    unseen_file.write_text("\n".join(sorted(unseen_drugs)) + "\n", encoding="utf-8")

    save_json(
        {
            "dataset": args.dataset,
            "mode": "inductive",
            "seed": int(args.seed),
            "requested_unseen_drug_ratio": float(args.unseen_ratio),
            "actual_unseen_drug_ratio": float(len(unseen_drugs) / len(candidate_drugs)),
            "num_seen_drugs": int(len(seen_drugs)),
            "num_unseen_drugs": int(len(unseen_drugs)),
            "seen_seen_validation_ratio": float(args.inductive_val_ratio),
            "partition_resampling_attempt": int(split_attempt),
            "negative_ratio": 1,
            "similarity_uses_all_drug_smiles_but_no_ddi_labels": True,
            "counts": counts,
        },
        f"split_meta_inductive_seed{args.seed}.json",
        args,
    )


def generate_splits(args: argparse.Namespace) -> None:
    positive_df, candidate_drugs, similarity_dim, relation_map = (
        load_valid_positive_triplets(args)
    )
    update_dataset_metadata(
        args=args,
        positive_df=positive_df,
        similarity_dim=similarity_dim,
        relation_map=relation_map,
    )

    if args.mode in {"transductive", "both"}:
        generate_transductive_splits(args, positive_df, candidate_drugs)
    if args.mode in {"inductive", "both"}:
        generate_inductive_splits(args, positive_df, candidate_drugs)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare paper-aligned SSE-DDI molecular data and splits."
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["drugbank", "twosides"],
    )
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--output-root", default="./data/processed")
    parser.add_argument(
        "--operation",
        choices=["all", "drug_data", "split"],
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=["transductive", "inductive", "both"],
        default="both",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--head-id-col", default=None)
    parser.add_argument("--tail-id-col", default=None)
    parser.add_argument("--head-smiles-col", default=None)
    parser.add_argument("--tail-smiles-col", default=None)
    parser.add_argument("--relation-col", default=None)
    parser.add_argument("--delimiter", default=None)

    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-nbits", type=int, default=2048)
    parser.add_argument("--sim-topk", type=int, default=32)
    parser.add_argument("--sim-quantile", type=float, default=0.70)
    parser.add_argument("--sim-std-lambda", type=float, default=0.50)
    parser.add_argument("--d-min", type=int, default=8)
    parser.add_argument("--d-max", type=int, default=64)

    parser.add_argument("--unseen-ratio", type=float, default=0.20)
    parser.add_argument("--inductive-val-ratio", type=float, default=0.20)
    parser.add_argument("--max-inductive-split-attempts", type=int, default=1000)
    parser.add_argument("--max-negative-trials", type=int, default=10000)
    parser.add_argument(
        "--allow-smiles-conflicts",
        action="store_true",
        help="Select the first canonical SMILES when one drug ID has conflicts.",
    )

    return parser


def configure_dataset_columns(args: argparse.Namespace) -> None:
    defaults = {
        "drugbank": {
            "head_id": "ID1",
            "tail_id": "ID2",
            "head_smiles": "X1",
            "tail_smiles": "X2",
            "relation": "Y",
            "delimiter": "\t",
        },
        "twosides": {
            "head_id": "Drug1_ID",
            "tail_id": "Drug2_ID",
            "head_smiles": "Drug1",
            "tail_smiles": "Drug2",
            "relation": "New Y",
            "delimiter": ",",
        },
    }[args.dataset]

    args.head_id_col = args.head_id_col or defaults["head_id"]
    args.tail_id_col = args.tail_id_col or defaults["tail_id"]
    args.head_smiles_col = args.head_smiles_col or defaults["head_smiles"]
    args.tail_smiles_col = args.tail_smiles_col or defaults["tail_smiles"]
    args.relation_col = args.relation_col or defaults["relation"]
    args.delimiter = decode_delimiter(args.delimiter or defaults["delimiter"])


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0.0 < args.sim_quantile < 1.0:
        raise ValueError("--sim-quantile must be between 0 and 1")
    if args.sim_std_lambda < 0.0:
        raise ValueError("--sim-std-lambda must be non-negative")
    if args.sim_topk < 1:
        raise ValueError("--sim-topk must be positive")
    if args.d_min < 0:
        raise ValueError("--d-min must be non-negative")
    if args.d_min > args.sim_topk:
        raise ValueError("--d-min cannot exceed --sim-topk")
    if args.d_max < args.d_min:
        raise ValueError("--d-max must be greater than or equal to --d-min")
    if args.morgan_radius < 0:
        raise ValueError("--morgan-radius must be non-negative")
    if args.morgan_nbits < 1:
        raise ValueError("--morgan-nbits must be positive")
    if not 0.0 < args.unseen_ratio < 1.0:
        raise ValueError("--unseen-ratio must be between 0 and 1")
    if not 0.0 < args.inductive_val_ratio < 1.0:
        raise ValueError("--inductive-val-ratio must be between 0 and 1")
    if args.max_inductive_split_attempts < 1:
        raise ValueError("--max-inductive-split-attempts must be positive")
    if args.max_negative_trials < 1:
        raise ValueError("--max-negative-trials must be positive")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    configure_dataset_columns(args)
    validate_arguments(args)

    if args.operation in {"all", "drug_data"}:
        prepare_drug_graphs(args)
    if args.operation in {"all", "split"}:
        generate_splits(args)


if __name__ == "__main__":
    main()
