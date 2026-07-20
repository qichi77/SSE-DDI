from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

# Importing CustomData makes pickle deserialization reliable when the graph
# objects were serialized by data_pre.py executed as a module.
from data_pre import CustomData  # noqa: F401


Triplet = Tuple[str, str, int]
Sample = Tuple[str, str, int, float]

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "Drug1_ID",
    "Drug2_ID",
    "Y",
    "label",
)

SPLITS_BY_MODE: Mapping[str, Tuple[str, ...]] = {
    "transductive": ("train", "val", "test"),
    "inductive": ("train", "val", "s1", "s2"),
}


@dataclass(frozen=True)
class PreparedDatasetMetadata:
    """Validated metadata needed by the loader and training script."""

    dataset: str
    num_drugs: int
    num_relations: int
    similarity_dim: int
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class GraphStore:
    """Shared molecular graph dictionaries for all data splits."""

    pyg: Mapping[str, Data]
    dgl: Mapping[str, dgl.DGLGraph]
    drug_index: Mapping[str, int]
    metadata: PreparedDatasetMetadata


class DDITripletDataset(Dataset):
    """Fixed relation-conditioned binary DDI instances.

    The dataset stores only identifiers, relation IDs, and labels. Molecular
    graphs are resolved in the collator, so no graph preprocessing or negative
    sampling occurs during training.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        graph_store: GraphStore,
        split_name: str,
    ) -> None:
        self.split_name = str(split_name)
        self.graph_store = graph_store

        normalized = normalize_split_dataframe(dataframe)
        validate_single_split(
            normalized,
            split_name=self.split_name,
            graph_store=graph_store,
        )

        self._head_ids = normalized["Drug1_ID"].to_numpy(dtype=object, copy=True)
        self._tail_ids = normalized["Drug2_ID"].to_numpy(dtype=object, copy=True)
        self._relations = normalized["Y"].to_numpy(dtype=np.int64, copy=True)
        self._labels = normalized["label"].to_numpy(dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> Sample:
        return (
            str(self._head_ids[index]),
            str(self._tail_ids[index]),
            int(self._relations[index]),
            float(self._labels[index]),
        )


class DDICollator:
    """Create synchronized PyG and DGL batches for both drugs."""

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

    def __call__(
        self,
        samples: Sequence[Sample],
    ) -> Tuple[
        Batch,
        Batch,
        dgl.DGLGraph,
        dgl.DGLGraph,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not samples:
            raise ValueError("Cannot collate an empty DDI batch")

        head_pyg_graphs: List[Data] = []
        tail_pyg_graphs: List[Data] = []
        head_dgl_graphs: List[dgl.DGLGraph] = []
        tail_dgl_graphs: List[dgl.DGLGraph] = []
        relations: List[int] = []
        labels: List[float] = []

        for head_id, tail_id, relation, label in samples:
            try:
                head_pyg = self.graph_store.pyg[head_id]
                tail_pyg = self.graph_store.pyg[tail_id]
                head_dgl = self.graph_store.dgl[head_id]
                tail_dgl = self.graph_store.dgl[tail_id]
            except KeyError as error:
                raise KeyError(
                    f"Drug {error.args[0]!r} in a split has no prepared molecular graph"
                ) from error

            head_pyg_graphs.append(head_pyg)
            tail_pyg_graphs.append(tail_pyg)
            head_dgl_graphs.append(head_dgl)
            tail_dgl_graphs.append(tail_dgl)
            relations.append(int(relation))
            labels.append(float(label))

        # edge_index is the directed bond-state set. follow_batch creates the
        # edge_index_batch vector required for molecule-wise SSE pooling.
        head_batch = Batch.from_data_list(
            head_pyg_graphs,
            follow_batch=["edge_index"],
        )
        tail_batch = Batch.from_data_list(
            tail_pyg_graphs,
            follow_batch=["edge_index"],
        )

        validate_pyg_batch(head_batch, expected_batch_size=len(samples), side="head")
        validate_pyg_batch(tail_batch, expected_batch_size=len(samples), side="tail")

        head_dgl_batch = dgl.batch(head_dgl_graphs)
        tail_dgl_batch = dgl.batch(tail_dgl_graphs)

        relation_tensor = torch.tensor(relations, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        return (
            head_batch,
            tail_batch,
            head_dgl_batch,
            tail_dgl_batch,
            relation_tensor,
            label_tensor,
        )


class DrugDataLoader(DataLoader):
    """DataLoader using the synchronized SSE-DDI graph collator."""

    def __init__(
        self,
        dataset: DDITripletDataset,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset,
            collate_fn=DDICollator(dataset.graph_store),
            **kwargs,
        )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        return pickle.load(file)


def normalize_identifier(value: Any) -> str:
    return str(value).strip()


def normalize_split_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Split CSV is missing columns {missing}; found {list(dataframe.columns)}"
        )

    output = dataframe.loc[:, list(REQUIRED_COLUMNS)].copy()
    output["Drug1_ID"] = output["Drug1_ID"].map(normalize_identifier)
    output["Drug2_ID"] = output["Drug2_ID"].map(normalize_identifier)

    if output["Drug1_ID"].eq("").any() or output["Drug2_ID"].eq("").any():
        raise ValueError("Split CSV contains an empty drug identifier")

    relation_numeric = pd.to_numeric(output["Y"], errors="raise")
    if not np.all(np.equal(relation_numeric, np.floor(relation_numeric))):
        raise ValueError("Relation IDs must be integers")
    output["Y"] = relation_numeric.astype(np.int64)

    label_numeric = pd.to_numeric(output["label"], errors="raise")
    if not np.all(np.isin(label_numeric.to_numpy(), [0, 1])):
        raise ValueError("Labels must be binary values 0 or 1")
    output["label"] = label_numeric.astype(np.int64)

    if output.empty:
        raise ValueError("Split CSV contains no instances")

    return output.reset_index(drop=True)


def triplet_set(dataframe: pd.DataFrame) -> Set[Triplet]:
    return set(
        zip(
            dataframe["Drug1_ID"].astype(str),
            dataframe["Drug2_ID"].astype(str),
            dataframe["Y"].astype(int),
        )
    )


def largest_remainder_counts(total: int, ratios: Sequence[float]) -> List[int]:
    """Match the integer allocation used by the paper-aligned data_pre.py."""

    if total < 1:
        raise ValueError("total must be positive")
    if not np.isclose(sum(ratios), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("ratios must sum to one")

    raw = np.asarray(ratios, dtype=np.float64) * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    fractional = raw - counts
    order = np.lexsort((np.arange(len(ratios)), -fractional))

    for index in order[:remainder]:
        counts[index] += 1

    return counts.astype(int).tolist()


def validate_single_split(
    dataframe: pd.DataFrame,
    split_name: str,
    graph_store: GraphStore,
) -> None:
    metadata = graph_store.metadata

    invalid_relation = (dataframe["Y"] < 0) | (
        dataframe["Y"] >= metadata.num_relations
    )
    if bool(invalid_relation.any()):
        values = sorted(dataframe.loc[invalid_relation, "Y"].unique().tolist())
        raise ValueError(
            f"{split_name} contains relation IDs outside [0, "
            f"{metadata.num_relations - 1}]: {values[:10]}"
        )

    graph_ids = set(graph_store.pyg)
    referenced_ids = set(dataframe["Drug1_ID"]) | set(dataframe["Drug2_ID"])
    missing_graphs = referenced_ids - graph_ids
    if missing_graphs:
        raise ValueError(
            f"{split_name} references drugs without graphs: {sorted(missing_graphs)[:10]}"
        )

    positive_count = int((dataframe["label"] == 1).sum())
    negative_count = int((dataframe["label"] == 0).sum())
    if positive_count != negative_count:
        raise ValueError(
            f"{split_name} violates the paper's 1:1 positive/negative ratio: "
            f"{positive_count}/{negative_count}"
        )

    positives = dataframe.loc[dataframe["label"] == 1]
    negatives = dataframe.loc[dataframe["label"] == 0]

    if positives.duplicated(subset=["Drug1_ID", "Drug2_ID", "Y"]).any():
        raise ValueError(f"{split_name} contains duplicate positive triplets")
    if negatives.duplicated(subset=["Drug1_ID", "Drug2_ID", "Y"]).any():
        raise ValueError(f"{split_name} contains duplicate negative triplets")

    if triplet_set(positives) & triplet_set(negatives):
        raise ValueError(f"{split_name} contains a triplet with conflicting labels")


def validate_global_split_protocol(
    frames: Mapping[str, pd.DataFrame],
    mode: str,
    graph_store: GraphStore,
    unseen_drugs: Optional[Set[str]],
) -> None:
    positive_sets: Dict[str, Set[Triplet]] = {}
    negative_sets: Dict[str, Set[Triplet]] = {}

    for split_name, dataframe in frames.items():
        validate_single_split(dataframe, split_name, graph_store)
        positive_sets[split_name] = triplet_set(
            dataframe.loc[dataframe["label"] == 1]
        )
        negative_sets[split_name] = triplet_set(
            dataframe.loc[dataframe["label"] == 0]
        )

    split_names = list(frames)
    for first_index, first_name in enumerate(split_names):
        for second_name in split_names[first_index + 1 :]:
            positive_overlap = positive_sets[first_name] & positive_sets[second_name]
            if positive_overlap:
                raise ValueError(
                    f"Positive triplets overlap between {first_name} and "
                    f"{second_name}: {len(positive_overlap)}"
                )

            negative_overlap = negative_sets[first_name] & negative_sets[second_name]
            if negative_overlap:
                raise ValueError(
                    f"Negative triplets overlap between {first_name} and "
                    f"{second_name}: {len(negative_overlap)}"
                )

    all_positives = set().union(*positive_sets.values())
    all_negatives = set().union(*negative_sets.values())
    leakage = all_positives & all_negatives
    if leakage:
        raise ValueError(
            f"Generated negatives overlap the complete observed positive set: "
            f"{len(leakage)} triplets"
        )

    metadata_positive_count = graph_store.metadata.raw.get("num_positive_triplets")
    if metadata_positive_count is not None and len(all_positives) != int(
        metadata_positive_count
    ):
        raise ValueError(
            "Loaded positive count does not match dataset_meta.json: "
            f"loaded={len(all_positives)}, metadata={metadata_positive_count}"
        )

    if mode == "transductive":
        positive_counts = [
            len(positive_sets["train"]),
            len(positive_sets["val"]),
            len(positive_sets["test"]),
        ]
        expected = largest_remainder_counts(
            sum(positive_counts),
            (0.60, 0.20, 0.20),
        )
        if positive_counts != expected:
            raise ValueError(
                "Transductive positive split is not the fixed 60%/20%/20% "
                f"allocation: observed={positive_counts}, expected={expected}"
            )
        return

    if unseen_drugs is None:
        raise ValueError("Inductive loading requires an unseen-drug set")

    all_drugs = set(graph_store.pyg)
    if not unseen_drugs or unseen_drugs == all_drugs:
        raise ValueError("The inductive unseen-drug set is empty or contains all drugs")
    if not unseen_drugs.issubset(all_drugs):
        unknown = unseen_drugs - all_drugs
        raise ValueError(f"Unseen-drug file contains unknown IDs: {sorted(unknown)[:10]}")

    requested_ratio = float(
        graph_store.metadata.raw.get("inductive_unseen_ratio", 0.20)
    )
    # data_pre.py records the exact ratio in split metadata; the default paper
    # protocol is 20%. The count check uses nearest-integer allocation.
    expected_unseen_count = int(np.floor(len(all_drugs) * requested_ratio + 0.5))
    expected_unseen_count = min(max(expected_unseen_count, 1), len(all_drugs) - 1)
    if len(unseen_drugs) != expected_unseen_count:
        raise ValueError(
            "Inductive unseen-drug count is inconsistent with the requested protocol: "
            f"observed={len(unseen_drugs)}, expected={expected_unseen_count}, "
            f"ratio={requested_ratio}"
        )

    for split_name, dataframe in frames.items():
        head_unseen = dataframe["Drug1_ID"].isin(unseen_drugs)
        tail_unseen = dataframe["Drug2_ID"].isin(unseen_drugs)

        if split_name in {"train", "val"}:
            valid = ~head_unseen & ~tail_unseen
        elif split_name == "s1":
            valid = head_unseen ^ tail_unseen
        elif split_name == "s2":
            valid = head_unseen & tail_unseen
        else:
            raise ValueError(f"Unexpected inductive split name: {split_name}")

        if not bool(valid.all()):
            invalid_count = int((~valid).sum())
            raise ValueError(
                f"{split_name} contains {invalid_count} instances that violate "
                "its seen/unseen definition"
            )

    seen_seen_positive_total = len(positive_sets["train"]) + len(positive_sets["val"])
    split_metadata = graph_store.metadata.raw.get("active_split_metadata", {})
    validation_ratio = float(split_metadata.get("seen_seen_validation_ratio", 0.20))
    expected_train_val = largest_remainder_counts(
        seen_seen_positive_total,
        (1.0 - validation_ratio, validation_ratio),
    )
    observed_train_val = [
        len(positive_sets["train"]),
        len(positive_sets["val"]),
    ]
    if observed_train_val != expected_train_val:
        raise ValueError(
            "Inductive seen--seen train/validation allocation is inconsistent: "
            f"observed={observed_train_val}, expected={expected_train_val}"
        )


def validate_pyg_graph(
    drug_id: str,
    graph: Data,
    similarity_dim: int,
) -> None:
    required_attributes = (
        "x",
        "edge_index",
        "line_graph_edge_index",
        "edge_attr",
        "sim",
    )
    missing = [name for name in required_attributes if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"PyG graph {drug_id!r} is missing attributes {missing}")

    if graph.x.ndim != 2 or graph.x.size(0) < 1:
        raise ValueError(f"PyG graph {drug_id!r} has invalid atom features")
    if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
        raise ValueError(f"PyG graph {drug_id!r} has invalid edge_index shape")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.size(0) != graph.edge_index.size(1):
        raise ValueError(f"PyG graph {drug_id!r} has inconsistent bond features")
    if (
        graph.line_graph_edge_index.ndim != 2
        or graph.line_graph_edge_index.size(0) != 2
    ):
        raise ValueError(
            f"PyG graph {drug_id!r} has invalid line_graph_edge_index shape"
        )

    if graph.line_graph_edge_index.numel():
        minimum = int(graph.line_graph_edge_index.min().item())
        maximum = int(graph.line_graph_edge_index.max().item())
        if minimum < 0 or maximum >= graph.edge_index.size(1):
            raise ValueError(
                f"PyG graph {drug_id!r} has line-graph indices outside the "
                "directed bond-state range"
            )

    if graph.sim.ndim != 2 or graph.sim.size(0) != 1:
        raise ValueError(f"PyG graph {drug_id!r} must store sim with shape [1, D]")
    if graph.sim.size(1) != similarity_dim:
        raise ValueError(
            f"PyG graph {drug_id!r} similarity dimension is {graph.sim.size(1)}, "
            f"expected {similarity_dim}"
        )

    graph_identifier = normalize_identifier(getattr(graph, "id", drug_id))
    if graph_identifier != drug_id:
        raise ValueError(
            f"PyG graph dictionary key {drug_id!r} disagrees with graph.id "
            f"{graph_identifier!r}"
        )


def validate_dgl_graph(
    drug_id: str,
    graph: dgl.DGLGraph,
    corresponding_pyg: Data,
) -> None:
    if "feat" not in graph.ndata:
        raise ValueError(f"DGL graph {drug_id!r} is missing ndata['feat']")
    if "feat" not in graph.edata:
        raise ValueError(f"DGL graph {drug_id!r} is missing edata['feat']")

    if graph.num_nodes() != corresponding_pyg.x.size(0):
        raise ValueError(f"PyG/DGL node-count mismatch for drug {drug_id!r}")
    if graph.num_edges() != corresponding_pyg.edge_index.size(1):
        raise ValueError(f"PyG/DGL directed-edge-count mismatch for drug {drug_id!r}")
    if graph.ndata["feat"].size(0) != graph.num_nodes():
        raise ValueError(f"DGL node-feature count mismatch for drug {drug_id!r}")
    if graph.edata["feat"].size(0) != graph.num_edges():
        raise ValueError(f"DGL edge-feature count mismatch for drug {drug_id!r}")


def validate_pyg_batch(batch: Batch, expected_batch_size: int, side: str) -> None:
    if not hasattr(batch, "edge_index_batch"):
        raise RuntimeError(
            f"The {side} PyG batch lacks edge_index_batch; SSE bond-state pooling "
            "cannot identify molecule membership"
        )
    if batch.edge_index_batch.numel() != batch.edge_index.size(1):
        raise RuntimeError(
            f"The {side} edge_index_batch length does not match the number of "
            "directed bond states"
        )
    if batch.sim.ndim != 2 or batch.sim.size(0) != expected_batch_size:
        raise RuntimeError(
            f"The {side} similarity batch must have shape [batch_size, D]"
        )
    if int(batch.num_graphs) != expected_batch_size:
        raise RuntimeError(
            f"The {side} PyG batch contains {batch.num_graphs} graphs, expected "
            f"{expected_batch_size}"
        )


def load_graph_store(
    root: Path,
    mode: str,
    seed: int,
    validate_graphs: bool,
) -> GraphStore:
    pyg_raw = read_pickle(root / "drug_data_pyg.pkl")
    dgl_raw = read_pickle(root / "drug_data_dgl.pkl")
    drug_index_raw = read_json(root / "drug_index.json")
    metadata_raw: MutableMapping[str, Any] = read_json(root / "dataset_meta.json")

    if not isinstance(pyg_raw, Mapping) or not isinstance(dgl_raw, Mapping):
        raise ValueError("Graph pickle files must contain drug-ID dictionaries")

    pyg_graphs: Dict[str, Data] = {
        normalize_identifier(drug_id): graph for drug_id, graph in pyg_raw.items()
    }
    dgl_graphs: Dict[str, dgl.DGLGraph] = {
        normalize_identifier(drug_id): graph for drug_id, graph in dgl_raw.items()
    }
    drug_index: Dict[str, int] = {
        normalize_identifier(drug_id): int(index)
        for drug_id, index in drug_index_raw.items()
    }

    if set(pyg_graphs) != set(dgl_graphs):
        raise ValueError("PyG and DGL graph dictionaries contain different drug IDs")
    if set(pyg_graphs) != set(drug_index):
        raise ValueError("Graph dictionaries and drug_index.json contain different IDs")

    expected_indices = set(range(len(drug_index)))
    if set(drug_index.values()) != expected_indices:
        raise ValueError("drug_index.json must be a one-to-one mapping onto 0..D-1")

    num_drugs = int(metadata_raw.get("num_drugs", len(pyg_graphs)))
    similarity_dim = int(metadata_raw.get("similarity_dim", len(pyg_graphs)))
    num_relations = int(metadata_raw.get("num_relations", 0))
    dataset_name = str(metadata_raw.get("dataset", root.name)).lower()

    if num_drugs != len(pyg_graphs):
        raise ValueError(
            f"dataset_meta.json num_drugs={num_drugs}, but loaded {len(pyg_graphs)} graphs"
        )
    if similarity_dim != len(pyg_graphs):
        raise ValueError(
            "The similarity profile must cover the complete benchmark drug dictionary: "
            f"similarity_dim={similarity_dim}, num_drugs={len(pyg_graphs)}"
        )
    if num_relations < 1:
        raise ValueError("dataset_meta.json must contain a positive num_relations")

    split_meta_path = root / f"split_meta_{mode}_seed{seed}.json"
    split_metadata = read_json(split_meta_path)
    if str(split_metadata.get("mode", "")).lower() != mode:
        raise ValueError(f"Mode mismatch in {split_meta_path}")
    if int(split_metadata.get("seed", -1)) != int(seed):
        raise ValueError(f"Seed mismatch in {split_meta_path}")
    if str(split_metadata.get("dataset", dataset_name)).lower() != dataset_name:
        raise ValueError(f"Dataset mismatch in {split_meta_path}")

    metadata_raw = dict(metadata_raw)
    metadata_raw["active_split_metadata"] = split_metadata
    if mode == "inductive":
        metadata_raw["inductive_unseen_ratio"] = float(
            split_metadata.get("requested_unseen_drug_ratio", 0.20)
        )

    metadata = PreparedDatasetMetadata(
        dataset=dataset_name,
        num_drugs=num_drugs,
        num_relations=num_relations,
        similarity_dim=similarity_dim,
        raw=metadata_raw,
    )

    if validate_graphs:
        for drug_id in sorted(pyg_graphs):
            validate_pyg_graph(drug_id, pyg_graphs[drug_id], similarity_dim)
            validate_dgl_graph(drug_id, dgl_graphs[drug_id], pyg_graphs[drug_id])

    return GraphStore(
        pyg=pyg_graphs,
        dgl=dgl_graphs,
        drug_index=drug_index,
        metadata=metadata,
    )


def read_split_csv(root: Path, mode: str, seed: int, split_name: str) -> pd.DataFrame:
    path = root / f"{mode}_seed{seed}_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Generate it with data_pre.py before training."
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "Drug1_ID": str,
            "Drug2_ID": str,
        },
        keep_default_na=False,
    )
    return normalize_split_dataframe(dataframe)


def read_unseen_drugs(root: Path, seed: int) -> Set[str]:
    path = root / f"inductive_seed{seed}_unseen_drugs.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    unseen = {
        normalize_identifier(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if normalize_identifier(line)
    }
    if not unseen:
        raise ValueError(f"{path} contains no unseen drug IDs")
    return unseen


def seed_worker(worker_id: int) -> None:
    """Make NumPy and Python RNGs deterministic in each DataLoader worker."""

    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataframe: pd.DataFrame,
    graph_store: GraphStore,
    split_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    prefetch_factor: Optional[int],
) -> DrugDataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = DDITripletDataset(
        dataframe=dataframe,
        graph_store=graph_store,
        split_name=split_name,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        # Do not discard the last partial batch; the paper does not define
        # sample dropping and every fixed instance should be used.
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
    }

    if num_workers > 0 and prefetch_factor is not None:
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DrugDataLoader(dataset, **loader_kwargs)


def load_ddi_dataset(
    root: str | Path,
    batch_size: int,
    mode: str,
    seed: int,
    num_workers: int = 0,
    *,
    validate_protocol: bool = True,
    validate_graphs: bool = True,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = 2,
) -> Dict[str, DrugDataLoader]:
    """Load fixed SSE-DDI splits and return named DataLoaders.

    Parameters
    ----------
    root:
        Processed dataset directory, for example
        ``data/processed/drugbank``.
    batch_size:
        Number of binary DDI instances per batch. The manuscript default is
        256, but the value remains a training configuration rather than a
        property of the dataset.
    mode:
        ``"transductive"`` or ``"inductive"``.
    seed:
        Selects the already generated split files. It does not create a new
        split and does not regenerate negatives.
    num_workers:
        PyTorch DataLoader worker count.
    validate_protocol:
        Check 1:1 balance, split disjointness, negative uniqueness, the
        60/20/20 allocation, and inductive S1/S2 constraints before training.
    validate_graphs:
        Check PyG/DGL graph keys, dimensions, and similarity-vector sizes.
    pin_memory:
        Forwarded to PyTorch DataLoader. It is false by default because batches
        include DGL graph objects.
    prefetch_factor:
        Applied only when ``num_workers > 0``.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    mode_normalized = str(mode).lower()
    if mode_normalized not in SPLITS_BY_MODE:
        raise ValueError(
            f"mode must be one of {sorted(SPLITS_BY_MODE)}, got {mode!r}"
        )

    graph_store = load_graph_store(
        root=root_path,
        mode=mode_normalized,
        seed=int(seed),
        validate_graphs=validate_graphs,
    )

    frames: Dict[str, pd.DataFrame] = {
        split_name: read_split_csv(
            root=root_path,
            mode=mode_normalized,
            seed=int(seed),
            split_name=split_name,
        )
        for split_name in SPLITS_BY_MODE[mode_normalized]
    }

    unseen_drugs = (
        read_unseen_drugs(root_path, int(seed))
        if mode_normalized == "inductive"
        else None
    )

    if validate_protocol:
        validate_global_split_protocol(
            frames=frames,
            mode=mode_normalized,
            graph_store=graph_store,
            unseen_drugs=unseen_drugs,
        )

    loaders: Dict[str, DrugDataLoader] = {}
    for split_name, dataframe in frames.items():
        loaders[split_name] = build_loader(
            dataframe=dataframe,
            graph_store=graph_store,
            split_name=split_name,
            batch_size=int(batch_size),
            shuffle=(split_name == "train"),
            num_workers=int(num_workers),
            seed=int(seed),
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        positives = int((dataframe["label"] == 1).sum())
        negatives = int((dataframe["label"] == 0).sum())
        print(
            f"Loaded {graph_store.metadata.dataset}/{mode_normalized}/"
            f"seed{seed}/{split_name}: {len(dataframe)} instances "
            f"({positives} positive, {negatives} negative)"
        )

    return loaders


__all__ = [
    "DDITripletDataset",
    "DrugDataLoader",
    "GraphStore",
    "PreparedDatasetMetadata",
    "build_loader",
    "load_ddi_dataset",
    "load_graph_store",
    "normalize_split_dataframe",
    "validate_global_split_protocol",
]
