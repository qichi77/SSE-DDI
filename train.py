from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# Importing CustomData keeps torch/pickle deserialization compatible with
# graph objects produced by data_pre.py.
from data_pre import CustomData  # noqa: F401
from dataset import DrugDataLoader, load_ddi_dataset
from model import gnn_model


METRIC_NAMES: Tuple[str, ...] = (
    "acc",
    "auc",
    "f1",
    "precision",
    "recall",
    "ap",
)

PAPER_DEFAULTS: Mapping[str, Any] = {
    "batch_size": 256,
    "weight_decay": 1e-3,
    "dropout": 0.2,
    "lr_gamma": 0.98,
    "hidden_dim": 96,
    "n_heads": 6,
    "transformer_layers": 2,
    "classification_threshold": 0.5,
    "transductive_learning_rate": 1e-4,
    "transductive_n_iter": 8,
    "inductive_learning_rate": 2e-5,
    "inductive_n_iter": 6,
}


class WeightedAverage:
    """Sample-weighted running average."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, sample_count: int) -> None:
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        self.total += float(value) * int(sample_count)
        self.count += int(sample_count)

    @property
    def average(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and DGL where available."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Required by some CUDA deterministic matrix-multiplication paths.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        import dgl

        dgl.seed(seed)
        if hasattr(dgl, "random"):
            dgl.random.seed(seed)
    except ImportError:
        # DGL is required by the actual model, but keeping this guard makes
        # static inspection and lightweight tests possible.
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(device_argument: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda``, or a concrete CUDA device."""

    requested = str(device_argument).strip().lower()

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_argument!r} was requested, but CUDA is unavailable"
        )

    return device


def compute_binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the six metrics reported in the manuscript."""

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if probabilities.size == 0:
        raise ValueError("Cannot compute metrics on an empty prediction array")
    if probabilities.shape != labels.shape:
        raise ValueError(
            "Prediction and label shapes differ: "
            f"{probabilities.shape} versus {labels.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities contain NaN or infinity")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(np.int64)

    if np.unique(labels).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probabilities))

    if int(np.sum(labels == 1)) == 0:
        ap = float("nan")
    else:
        ap = float(average_precision_score(labels, probabilities))

    return {
        "acc": float(accuracy_score(labels, predictions)),
        "auc": auc,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "ap": ap,
    }


def move_batch_to_device(
    batch: Tuple[Any, Any, Any, Any, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[Any, Any, Any, Any, torch.Tensor, torch.Tensor]:
    """Move synchronized PyG, DGL, relation, and label objects to one device."""

    (
        head_pyg,
        tail_pyg,
        head_dgl,
        tail_dgl,
        relation,
        labels,
    ) = batch

    head_pyg = head_pyg.to(device)
    tail_pyg = tail_pyg.to(device)
    head_dgl = head_dgl.to(device)
    tail_dgl = tail_dgl.to(device)
    relation = relation.to(device=device, dtype=torch.long)
    labels = labels.to(device=device, dtype=torch.float32)

    return (
        head_pyg,
        tail_pyg,
        head_dgl,
        tail_dgl,
        relation,
        labels,
    )


def forward_batch(
    model: nn.Module,
    batch: Tuple[Any, Any, Any, Any, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one synchronized DDI batch through SSE-DDI."""

    (
        head_pyg,
        tail_pyg,
        head_dgl,
        tail_dgl,
        relation,
        labels,
    ) = move_batch_to_device(batch, device)

    if "feat" not in head_dgl.edata or "feat" not in tail_dgl.edata:
        raise KeyError("DGL molecular graphs must contain edge feature key 'feat'")

    if not hasattr(head_pyg, "sim") or not hasattr(tail_pyg, "sim"):
        raise AttributeError("PyG molecular graphs must contain similarity feature 'sim'")

    logits = model(
        head_pyg,
        tail_pyg,
        head_dgl,
        tail_dgl,
        head_dgl.edata["feat"],
        tail_dgl.edata["feat"],
        relation,
        head_pyg.sim,
        tail_pyg.sim,
    )

    logits = logits.reshape(-1)
    labels = labels.reshape(-1)

    if logits.shape != labels.shape:
        raise RuntimeError(
            f"Model returned logits with shape {logits.shape}, "
            f"but labels have shape {labels.shape}"
        )

    return logits, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DrugDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: Optional[float],
    classification_threshold: float,
) -> Dict[str, float]:
    model.train()

    loss_meter = WeightedAverage()
    probability_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    progress = tqdm(
        dataloader,
        desc=f"train epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        logits, labels = forward_batch(model, batch, device)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss encountered at epoch {epoch}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(max_grad_norm),
            )

        optimizer.step()

        batch_size = int(labels.numel())
        loss_meter.update(float(loss.item()), batch_size)

        probability_parts.append(
            torch.sigmoid(logits).detach().cpu().numpy()
        )
        label_parts.append(labels.detach().cpu().numpy())

        progress.set_postfix(loss=f"{loss_meter.average:.6f}")

    probabilities = np.concatenate(probability_parts)
    labels = np.concatenate(label_parts)

    metrics = compute_binary_metrics(
        probabilities=probabilities,
        labels=labels,
        threshold=classification_threshold,
    )
    metrics["loss"] = float(loss_meter.average)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DrugDataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    classification_threshold: float,
) -> Dict[str, float]:
    model.eval()

    loss_meter = WeightedAverage()
    probability_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    for batch in tqdm(
        dataloader,
        desc=f"evaluate {split_name}",
        leave=False,
        dynamic_ncols=True,
    ):
        logits, labels = forward_batch(model, batch, device)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite evaluation loss encountered on {split_name}"
            )

        batch_size = int(labels.numel())
        loss_meter.update(float(loss.item()), batch_size)

        probability_parts.append(torch.sigmoid(logits).cpu().numpy())
        label_parts.append(labels.cpu().numpy())

    probabilities = np.concatenate(probability_parts)
    labels = np.concatenate(label_parts)

    metrics = compute_binary_metrics(
        probabilities=probabilities,
        labels=labels,
        threshold=classification_threshold,
    )
    metrics["loss"] = float(loss_meter.average)
    metrics["num_instances"] = int(labels.size)
    metrics["num_positive"] = int(np.sum(labels == 1))
    metrics["num_negative"] = int(np.sum(labels == 0))
    return metrics


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def choose_validation_score(
    metrics: Mapping[str, float],
    selection_metric: str,
) -> Tuple[float, str]:
    """Return a score to maximize and the metric actually used."""

    selection_metric = str(selection_metric).lower()

    if selection_metric == "last":
        return 0.0, "last"

    if selection_metric == "loss":
        loss = float(metrics["loss"])
        if not math.isfinite(loss):
            raise ValueError("Validation loss is not finite")
        return -loss, "loss"

    preferred_order = (
        [selection_metric]
        if selection_metric in {"auc", "ap"}
        else []
    )

    # Fallbacks make checkpoint selection robust when a pathological validation
    # subset contains only one label class.
    for metric_name in preferred_order + ["auc", "ap"]:
        value = float(metrics.get(metric_name, float("nan")))
        if math.isfinite(value):
            return value, metric_name

    loss = float(metrics["loss"])
    if not math.isfinite(loss):
        raise ValueError("No finite validation selection metric is available")
    return -loss, "loss"


def infer_graph_dimensions(
    train_loader: DrugDataLoader,
) -> Tuple[int, int, Any]:
    graph_store = train_loader.dataset.graph_store
    if not graph_store.pyg:
        raise ValueError("The prepared molecular graph store is empty")

    first_graph = next(iter(graph_store.pyg.values()))

    if not hasattr(first_graph, "x") or first_graph.x.ndim != 2:
        raise ValueError("Prepared PyG graphs must contain a 2-D x tensor")
    if not hasattr(first_graph, "edge_attr") or first_graph.edge_attr.ndim != 2:
        raise ValueError("Prepared PyG graphs must contain a 2-D edge_attr tensor")

    node_feature_dim = int(first_graph.x.size(-1))
    edge_feature_dim = int(first_graph.edge_attr.size(-1))

    if node_feature_dim < 1 or edge_feature_dim < 1:
        raise ValueError("Node and edge feature dimensions must be positive")

    return node_feature_dim, edge_feature_dim, graph_store.metadata


def mode_specific_hyperparameters(
    args: argparse.Namespace,
) -> Tuple[float, int]:
    if args.mode == "transductive":
        default_lr = float(PAPER_DEFAULTS["transductive_learning_rate"])
        default_n_iter = int(PAPER_DEFAULTS["transductive_n_iter"])
    else:
        default_lr = float(PAPER_DEFAULTS["inductive_learning_rate"])
        default_n_iter = int(PAPER_DEFAULTS["inductive_n_iter"])

    learning_rate = (
        default_lr if args.learning_rate is None else float(args.learning_rate)
    )
    n_iter = default_n_iter if args.n_iter is None else int(args.n_iter)
    return learning_rate, n_iter


def validate_paper_hyperparameters(
    args: argparse.Namespace,
    learning_rate: float,
    n_iter: int,
) -> None:
    """Reject silent deviations when strict paper mode is active."""

    if not args.strict_paper_hparams:
        return

    expected_lr = float(
        PAPER_DEFAULTS[
            "transductive_learning_rate"
            if args.mode == "transductive"
            else "inductive_learning_rate"
        ]
    )
    expected_n_iter = int(
        PAPER_DEFAULTS[
            "transductive_n_iter"
            if args.mode == "transductive"
            else "inductive_n_iter"
        ]
    )

    checks = {
        "batch_size": (int(args.batch_size), int(PAPER_DEFAULTS["batch_size"])),
        "weight_decay": (
            float(args.weight_decay),
            float(PAPER_DEFAULTS["weight_decay"]),
        ),
        "dropout": (float(args.dropout), float(PAPER_DEFAULTS["dropout"])),
        "lr_gamma": (float(args.lr_gamma), float(PAPER_DEFAULTS["lr_gamma"])),
        "hidden_dim": (
            int(args.hidden_dim),
            int(PAPER_DEFAULTS["hidden_dim"]),
        ),
        "n_heads": (int(args.n_heads), int(PAPER_DEFAULTS["n_heads"])),
        "transformer_layers": (
            int(args.transformer_layers),
            int(PAPER_DEFAULTS["transformer_layers"]),
        ),
        "classification_threshold": (
            float(args.classification_threshold),
            float(PAPER_DEFAULTS["classification_threshold"]),
        ),
        "learning_rate": (float(learning_rate), expected_lr),
        "n_iter": (int(n_iter), expected_n_iter),
    }

    mismatches: List[str] = []
    for name, (actual, expected) in checks.items():
        if isinstance(expected, float):
            equal = math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        else:
            equal = actual == expected

        if not equal:
            mismatches.append(f"{name}: actual={actual}, paper={expected}")

    if mismatches:
        raise ValueError(
            "Strict paper hyperparameter validation failed:\n  "
            + "\n  ".join(mismatches)
        )


def build_net_params(
    args: argparse.Namespace,
    device: torch.device,
    node_feature_dim: int,
    edge_feature_dim: int,
    num_relations: int,
    similarity_dim: int,
    n_iter: int,
) -> Dict[str, Any]:
    if args.hidden_dim % args.n_heads != 0:
        raise ValueError("hidden_dim must be divisible by n_heads")

    return {
        "L": int(args.transformer_layers),
        "n_heads": int(args.n_heads),
        "hidden_dim": int(args.hidden_dim),
        "out_dim": int(args.hidden_dim),
        "edge_feat": True,
        "residual": True,
        "readout": "max_mean",
        "in_feat_dropout": float(args.dropout),
        "dropout": float(args.dropout),
        "layer_norm": False,
        "batch_norm": True,
        "self_loop": False,
        "lap_pos_enc": False,
        "pos_enc_dim": 0,
        "full_graph": False,
        "batch_size": int(args.batch_size),
        "num_atom_type": int(node_feature_dim),
        "num_bond_type": int(edge_feature_dim),
        "device": device,
        "n_iter": int(n_iter),
        "num_relations": int(num_relations),
        # Current model.py uses sim_dim. similarity_dim is included as a
        # compatibility alias for alternative implementations.
        "sim_dim": int(similarity_dim),
        "similarity_dim": int(similarity_dim),
        "collect_routes": False,
    }


def validate_model_dimensions(
    model: nn.Module,
    num_relations: int,
    similarity_dim: int,
) -> None:
    relation_module = getattr(model, "rmodule", None)
    if relation_module is not None:
        actual = int(relation_module.num_embeddings)
        if actual != int(num_relations):
            raise ValueError(
                "Relation embedding size mismatch: "
                f"model={actual}, dataset={num_relations}"
            )

    similarity_layer = getattr(model, "lin_sim", None)
    if similarity_layer is not None:
        actual = int(similarity_layer.in_features)
        if actual != int(similarity_dim):
            raise ValueError(
                "Similarity input size mismatch: "
                f"model={actual}, dataset={similarity_dim}"
            )


def build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(args.weight_decay),
        )

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def format_metrics(metrics: Mapping[str, Any]) -> str:
    ordered_names = ["loss", *METRIC_NAMES]
    parts: List[str] = []

    for name in ordered_names:
        if name not in metrics:
            continue
        value = metrics[name]
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                parts.append(f"{name}={float(value):.6f}")
            else:
                parts.append(f"{name}=nan")

    return ", ".join(parts)


def make_json_safe(value: Any) -> Any:
    """Convert tensors, NumPy values, paths, and non-finite floats for JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(data),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
    }

    try:
        import dgl

        versions["dgl"] = getattr(dgl, "__version__", None)
    except ImportError:
        versions["dgl"] = None

    try:
        import torch_geometric

        versions["torch_geometric"] = getattr(
            torch_geometric,
            "__version__",
            None,
        )
    except ImportError:
        versions["torch_geometric"] = None

    return versions


def train_one_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    run_started = time.time()
    set_global_seed(seed, deterministic=args.deterministic)

    learning_rate, n_iter = mode_specific_hyperparameters(args)
    validate_paper_hyperparameters(args, learning_rate, n_iter)

    dataset_root = Path(args.data_root) / args.dataset
    loaders = load_ddi_dataset(
        root=dataset_root,
        batch_size=int(args.batch_size),
        mode=args.mode,
        seed=int(seed),
        num_workers=int(args.num_workers),
        validate_protocol=bool(args.validate_protocol),
        validate_graphs=bool(args.validate_graphs),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor),
    )

    node_feature_dim, edge_feature_dim, metadata = infer_graph_dimensions(
        loaders["train"]
    )

    if metadata.dataset.lower() != args.dataset.lower():
        raise ValueError(
            f"Dataset metadata says {metadata.dataset!r}, "
            f"but --dataset is {args.dataset!r}"
        )

    net_params = build_net_params(
        args=args,
        device=device,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        num_relations=metadata.num_relations,
        similarity_dim=metadata.similarity_dim,
        n_iter=n_iter,
    )

    model = gnn_model("GraphTransformer", net_params).to(device)
    validate_model_dimensions(
        model=model,
        num_relations=metadata.num_relations,
        similarity_dim=metadata.similarity_dim,
    )

    optimizer = build_optimizer(args, model, learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(args.lr_gamma),
    )
    criterion = nn.BCEWithLogitsLoss()

    best_score = -math.inf
    best_epoch = -1
    best_metric_used = ""
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
            classification_threshold=float(args.classification_threshold),
        )
        validation_metrics = evaluate(
            model=model,
            dataloader=loaders["val"],
            criterion=criterion,
            device=device,
            split_name="validation",
            classification_threshold=float(args.classification_threshold),
        )

        if args.selection_metric == "last":
            should_select = epoch == int(args.epochs)
            selection_score = float(epoch)
            metric_used = "last"
        else:
            selection_score, metric_used = choose_validation_score(
                validation_metrics,
                args.selection_metric,
            )
            should_select = selection_score > best_score

        if should_select:
            best_score = float(selection_score)
            best_epoch = int(epoch)
            best_metric_used = metric_used
            best_state = state_dict_to_cpu(model)

        history.append(
            {
                "epoch": int(epoch),
                "learning_rate": epoch_learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_score": float(selection_score),
                "selection_metric_used": metric_used,
            }
        )

        print(
            f"Seed {seed} | epoch {epoch:03d}/{args.epochs} | "
            f"lr={epoch_learning_rate:.8g} | "
            f"train: {format_metrics(train_metrics)} | "
            f"validation: {format_metrics(validation_metrics)}"
        )

        scheduler.step()

    if best_state is None or best_epoch < 1:
        raise RuntimeError("No checkpoint was selected")

    model.load_state_dict(best_state)
    model.to(device)

    checkpoint_directory = (
        Path(args.checkpoint_root) / args.dataset / args.mode
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"seed{seed}_best.pt"

    final_evaluation: Dict[str, Dict[str, float]] = {}
    if args.mode == "transductive":
        final_evaluation["test"] = evaluate(
            model=model,
            dataloader=loaders["test"],
            criterion=criterion,
            device=device,
            split_name="test",
            classification_threshold=float(args.classification_threshold),
        )
    else:
        final_evaluation["s1"] = evaluate(
            model=model,
            dataloader=loaders["s1"],
            criterion=criterion,
            device=device,
            split_name="S1",
            classification_threshold=float(args.classification_threshold),
        )
        final_evaluation["s2"] = evaluate(
            model=model,
            dataloader=loaders["s2"],
            criterion=criterion,
            device=device,
            split_name="S2",
            classification_threshold=float(args.classification_threshold),
        )

    checkpoint_payload = {
        "model_state_dict": best_state,
        "net_params": {
            key: value
            for key, value in net_params.items()
            if key != "device"
        },
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "selection_metric_requested": args.selection_metric,
        "selection_metric_used": best_metric_used,
        "selection_score": float(best_score),
        "classification_threshold": float(args.classification_threshold),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": args.optimizer,
            "initial_learning_rate": float(learning_rate),
            "weight_decay": float(args.weight_decay),
            "lr_scheduler": "ExponentialLR",
            "lr_gamma": float(args.lr_gamma),
            "dropout": float(args.dropout),
            "hidden_dim": int(args.hidden_dim),
            "n_heads": int(args.n_heads),
            "transformer_layers": int(args.transformer_layers),
            "n_iter": int(n_iter),
            "max_grad_norm": args.max_grad_norm,
        },
        "dataset_metadata": metadata.raw,
        "package_versions": package_versions(),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    result: Dict[str, Any] = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": int(seed),
        "device": str(device),
        "deterministic": bool(args.deterministic),
        "best_epoch": int(best_epoch),
        "selection_metric_requested": args.selection_metric,
        "selection_metric_used": best_metric_used,
        "selection_score": float(best_score),
        "classification_threshold": float(args.classification_threshold),
        "checkpoint": str(checkpoint_path),
        "hyperparameters": checkpoint_payload["hyperparameters"],
        "paper_hyperparameter_validation": bool(args.strict_paper_hparams),
        "dataset_metadata": metadata.raw,
        "final_evaluation": final_evaluation,
        "duration_seconds": float(time.time() - run_started),
        "package_versions": package_versions(),
    }

    if args.save_history:
        result["history"] = history

    # Retain convenient top-level scenario keys for downstream scripts.
    result.update(final_evaluation)

    result_path = (
        Path(args.result_root)
        / args.dataset
        / args.mode
        / f"seed{seed}.json"
    )
    save_json(result, result_path)

    for scenario, scenario_metrics in final_evaluation.items():
        print(
            f"Seed {seed} | {scenario.upper()}: "
            f"{format_metrics(scenario_metrics)}"
        )

    print(f"Seed {seed} | checkpoint: {checkpoint_path}")
    print(f"Seed {seed} | result: {result_path}")

    return result


def aggregate_results(
    seed_results: Sequence[Mapping[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    if not seed_results:
        raise ValueError("No seed results were provided")

    scenarios = ("test",) if mode == "transductive" else ("s1", "s2")
    summary: Dict[str, Any] = {}

    for scenario in scenarios:
        summary[scenario] = {}

        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    float(result[scenario][metric_name])
                    for result in seed_results
                ],
                dtype=np.float64,
            )

            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                mean = float("nan")
                std = float("nan")
            else:
                mean = float(np.mean(finite_values))
                std = (
                    float(np.std(finite_values, ddof=1))
                    if finite_values.size > 1
                    else float("nan")
                )

            summary[scenario][metric_name] = {
                "mean": mean,
                "std": std,
                "mean_percent": 100.0 * mean if math.isfinite(mean) else float("nan"),
                "std_percent": 100.0 * std if math.isfinite(std) else float("nan"),
                "values": values.tolist(),
                "num_finite_runs": int(finite_values.size),
            }

    return summary


def validate_arguments(args: argparse.Namespace) -> None:
    args.dataset = args.dataset.lower()
    args.mode = args.mode.lower()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be positive")
    if args.n_heads < 1:
        raise ValueError("--n-heads must be positive")
    if args.transformer_layers < 1:
        raise ValueError("--transformer-layers must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("--lr-gamma must be in (0, 1]")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.n_iter is not None and args.n_iter < 1:
        raise ValueError("--n-iter must be positive")
    if not 0.0 <= args.classification_threshold <= 1.0:
        raise ValueError("--classification-threshold must be between 0 and 1")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")

    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("--seeds must contain at least one value")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicate values")
    args.seeds = seeds

    if args.strict_paper_hparams and len(seeds) != 5:
        raise ValueError(
            "The manuscript reports five independent runs. "
            "Strict paper mode therefore requires exactly five distinct seeds."
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate paper-aligned SSE-DDI."
    )

    parser.add_argument(
        "--dataset",
        choices=["drugbank", "twosides"],
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=["transductive", "inductive"],
        required=True,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help=(
            "Prepared split seeds to run. The manuscript states five runs "
            "but does not list the exact seed values."
        ),
    )

    parser.add_argument("--data-root", default="./data/processed")
    parser.add_argument("--result-root", default="./results")
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    parser.add_argument("--device", default="auto")

    # The manuscript does not state the epoch count. This explicit repository
    # default must be added to the manuscript if used for final results.
    parser.add_argument("--epochs", type=int, default=200)

    # Manuscript-defined defaults.
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr-gamma", type=float, default=0.98)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--classification-threshold", type=float, default=0.5)

    # The optimizer and checkpoint rule are not stated in the manuscript.
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument(
        "--selection-metric",
        choices=["auc", "ap", "loss", "last"],
        default="auc",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-paper-hparams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject changes to manuscript-defined hyperparameters and require "
            "exactly five seeds."
        ),
    )
    parser.add_argument(
        "--save-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-grad-norm", type=float, default=None)

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    seed_results: List[Dict[str, Any]] = []

    for seed in args.seeds:
        seed_results.append(
            train_one_seed(
                args=args,
                seed=int(seed),
                device=device,
            )
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_results(
        seed_results=seed_results,
        mode=args.mode,
    )

    summary_payload: Dict[str, Any] = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seeds": [int(seed) for seed in args.seeds],
        "num_runs": int(len(args.seeds)),
        "standard_deviation": "sample standard deviation (ddof=1)",
        "summary": summary,
        "paper_defined_defaults": dict(PAPER_DEFAULTS),
        "protocol_choices_not_explicit_in_manuscript": {
            "optimizer": args.optimizer,
            "epochs": int(args.epochs),
            "exact_seed_values": [int(seed) for seed in args.seeds],
            "checkpoint_selection": args.selection_metric,
        },
        "package_versions": package_versions(),
    }

    summary_path = (
        Path(args.result_root)
        / args.dataset
        / args.mode
        / "summary.json"
    )
    save_json(summary_payload, summary_path)

    print("\nFinal five-run summary")
    for scenario, scenario_metrics in summary.items():
        print(f"\n[{scenario.upper()}]")
        for metric_name in METRIC_NAMES:
            metric_summary = scenario_metrics[metric_name]
            mean_percent = metric_summary["mean_percent"]
            std_percent = metric_summary["std_percent"]

            if math.isfinite(mean_percent) and math.isfinite(std_percent):
                print(
                    f"{metric_name:10s}: "
                    f"{mean_percent:.4f} ± {std_percent:.4f}"
                )
            else:
                print(f"{metric_name:10s}: unavailable")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
