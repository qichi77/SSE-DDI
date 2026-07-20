from __future__ import annotations

import json
import math
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn


T = TypeVar("T")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its ``Path`` object."""

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = True,
) -> int:
    """Seed Python, NumPy, PyTorch, CUDA, and DGL when available."""

    seed = int(seed)

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )

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
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=bool(warn_only),
            )
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        try:
            torch.use_deterministic_algorithms(False)
        except (AttributeError, TypeError):
            pass

    return seed


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = True,
) -> int:
    """Alias used by the paper-aligned training script."""

    return set_seed(
        seed=seed,
        deterministic=deterministic,
        warn_only=warn_only,
    )


def seed_worker(worker_id: int) -> None:
    """Seed one PyTorch DataLoader worker."""

    del worker_id

    worker_seed = int(
        torch.initial_seed() % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Generator:
    """Create a seeded ``torch.Generator`` for deterministic DataLoaders."""

    generator = torch.Generator(
        device=torch.device(device)
    )
    generator.manual_seed(int(seed))
    return generator


class BestMeter:
    """Track the best scalar value and non-improving step count."""

    def __init__(self, best_type: str = "max") -> None:
        normalized = str(best_type).lower()

        if normalized not in {"min", "max"}:
            raise ValueError(
                "best_type must be 'min' or 'max'"
            )

        self.best_type = normalized
        self.reset()

    def reset(self) -> None:
        self.best = (
            float("inf")
            if self.best_type == "min"
            else -float("inf")
        )
        self.count = 0

    def is_better(self, value: float) -> bool:
        value = float(value)

        if not math.isfinite(value):
            return False

        if self.best_type == "min":
            return value < self.best

        return value > self.best

    def update(self, best: float) -> bool:
        """Update the stored value and return whether it improved."""

        value = float(best)

        if self.is_better(value):
            self.best = value
            self.count = 0
            return True

        self.count += 1
        return False

    def get_best(self) -> float:
        return float(self.best)

    def counter(self) -> int:
        """Legacy method: increment and return the counter."""

        self.count += 1
        return int(self.count)


class AverageMeter:
    """Sample-weighted running average compatible with the original code."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        val: float | torch.Tensor | np.generic,
        n: int = 1,
    ) -> None:
        if n < 0:
            raise ValueError("n cannot be negative")

        if torch.is_tensor(val):
            if val.numel() != 1:
                raise ValueError(
                    "AverageMeter accepts only scalar tensors"
                )
            value = float(val.detach().item())
        else:
            value = float(val)

        if not math.isfinite(value):
            raise ValueError(
                "AverageMeter cannot update with NaN or infinity"
            )

        self.val = value
        self.sum += value * int(n)
        self.count += int(n)

        if self.count > 0:
            self.avg = self.sum / self.count

    def get_average(self) -> float:
        if self.count == 0:
            return float("nan")

        self.avg = self.sum / self.count
        return float(self.avg)


def minmax_normalize(
    values: Any,
    eps: float = 1e-12,
) -> Any:
    """Perform numerically safe min-max normalization."""

    eps = float(eps)

    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be a positive finite number")

    if torch.is_tensor(values):
        if values.numel() == 0:
            raise ValueError(
                "Cannot normalize an empty tensor"
            )

        tensor = (
            values
            if torch.is_floating_point(values)
            else values.to(dtype=torch.float32)
        )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "Input tensor contains NaN or infinity"
            )

        minimum = tensor.min()
        maximum = tensor.max()
        span = maximum - minimum

        if float(span.abs().item()) <= eps:
            return torch.zeros_like(tensor)

        return (tensor - minimum) / span

    array = np.asarray(values)

    if array.size == 0:
        raise ValueError(
            "Cannot normalize an empty array"
        )

    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    else:
        array = array.astype(array.dtype, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Input array contains NaN or infinity"
        )

    minimum = np.min(array)
    maximum = np.max(array)
    span = float(maximum - minimum)

    if abs(span) <= eps:
        return np.zeros_like(array)

    return (array - minimum) / span


def normalize(
    values: Any,
    eps: float = 1e-12,
) -> Any:
    """Backward-compatible alias for safe min-max normalization."""

    return minmax_normalize(
        values=values,
        eps=eps,
    )


def state_dict_to_cpu(
    model_or_state_dict: nn.Module | Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Copy a model state dictionary to CPU for portable checkpointing."""

    if isinstance(model_or_state_dict, nn.Module):
        state_dict = model_or_state_dict.state_dict()
    else:
        state_dict = model_or_state_dict

    output: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"State value {key!r} is not a tensor"
            )
        output[str(key)] = (
            value.detach().cpu().clone()
        )

    return output


def make_json_safe(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


def atomic_torch_save(
    value: Any,
    path: str | Path,
) -> Path:
    """Atomically write a PyTorch checkpoint in the destination directory."""

    destination = Path(path).expanduser()
    ensure_directory(destination.parent)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def save_json(
    value: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Atomically save strict, UTF-8 JSON."""

    destination = Path(path).expanduser()
    ensure_directory(destination.parent)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                make_json_safe(value),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def save_checkpoint(
    model: nn.Module,
    model_dir: str | Path,
    epoch: int,
    val_loss: float,
    val_acc: float,
    *,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[
        Mapping[str, Any]
    ] = None,
    filename: Optional[str] = None,
) -> Path:
    """Save a self-describing, portable training checkpoint."""

    directory = ensure_directory(model_dir)

    if filename is None:
        filename = (
            f"epoch-{int(epoch):04d}_"
            f"val-loss-{float(val_loss):.6f}_"
            f"val-acc-{float(val_acc):.6f}.pt"
        )

    checkpoint_path = directory / filename

    payload: Dict[str, Any] = {
        "model_state_dict": state_dict_to_cpu(model),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
    }

    if optimizer is not None:
        payload[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if scheduler is not None:
        if not hasattr(
            scheduler,
            "state_dict",
        ):
            raise TypeError(
                "scheduler must implement state_dict()"
            )
        payload[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    if extra is not None:
        payload["extra"] = dict(extra)

    return atomic_torch_save(
        value=payload,
        path=checkpoint_path,
    )


def _safe_torch_load(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load both modern and older PyTorch checkpoints."""

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def load_checkpoint(
    model_path: str | Path,
    model: Optional[nn.Module] = None,
    *,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Any:
    """Load a checkpoint and optionally restore training objects."""

    checkpoint_path = Path(model_path).expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    payload = _safe_torch_load(
        checkpoint_path,
        map_location=map_location,
    )

    if model is None:
        return payload

    if isinstance(payload, nn.Module):
        model.load_state_dict(
            payload.state_dict(),
            strict=strict,
        )
        return payload

    if isinstance(payload, Mapping):
        if "model_state_dict" in payload:
            state_dict = payload[
                "model_state_dict"
            ]
        elif "state_dict" in payload:
            state_dict = payload[
                "state_dict"
            ]
        else:
            state_dict = payload

        model.load_state_dict(
            state_dict,
            strict=strict,
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in payload
        ):
            optimizer.load_state_dict(
                payload[
                    "optimizer_state_dict"
                ]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in payload
        ):
            scheduler.load_state_dict(
                payload[
                    "scheduler_state_dict"
                ]
            )

        return payload

    raise TypeError(
        f"Unsupported checkpoint type: "
        f"{type(payload).__name__}"
    )


def save_model_dict(
    model: nn.Module,
    model_dir: str | Path,
    msg: str,
) -> Path:
    """Save only a portable model state dictionary."""

    directory = ensure_directory(
        model_dir
    )
    path = directory / f"{msg}.pt"

    atomic_torch_save(
        state_dict_to_cpu(model),
        path,
    )

    print(
        f"Model state dictionary saved to {path}."
    )
    return path


def load_model_dict(
    model: nn.Module,
    ckpt: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load a model state dictionary and return the model."""

    payload = _safe_torch_load(
        ckpt,
        map_location=map_location,
    )

    if isinstance(payload, Mapping):
        if "model_state_dict" in payload:
            state_dict = payload[
                "model_state_dict"
            ]
        elif "state_dict" in payload:
            state_dict = payload[
                "state_dict"
            ]
        else:
            state_dict = payload
    elif isinstance(payload, nn.Module):
        state_dict = payload.state_dict()
    else:
        raise TypeError(
            f"Unsupported model checkpoint type: "
            f"{type(payload).__name__}"
        )

    model.load_state_dict(
        state_dict,
        strict=strict,
    )
    return model


def environment_info() -> Dict[str, Any]:
    """Return software information suitable for result metadata."""

    information: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": (
            torch.version.cuda
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cudnn": (
            str(
                torch.backends.cudnn.version()
            )
            if torch.backends.cudnn.is_available()
            else None
        ),
    }

    try:
        import dgl

        information["dgl"] = getattr(
            dgl,
            "__version__",
            None,
        )
    except ImportError:
        information["dgl"] = None

    try:
        import torch_geometric

        information[
            "torch_geometric"
        ] = getattr(
            torch_geometric,
            "__version__",
            None,
        )
    except ImportError:
        information[
            "torch_geometric"
        ] = None

    try:
        import rdkit

        information["rdkit"] = getattr(
            rdkit,
            "__version__",
            None,
        )
    except ImportError:
        information["rdkit"] = None

    return information


def cycle(iterable: Iterable[T]) -> Iterator[T]:
    """Yield elements from an iterable forever without printing."""

    while True:
        yielded = False

        for item in iterable:
            yielded = True
            yield item

        if not yielded:
            raise ValueError(
                "Cannot cycle an empty iterable"
            )


__all__ = [
    "AverageMeter",
    "BestMeter",
    "atomic_torch_save",
    "cycle",
    "ensure_directory",
    "environment_info",
    "load_checkpoint",
    "load_model_dict",
    "make_json_safe",
    "make_torch_generator",
    "minmax_normalize",
    "normalize",
    "save_checkpoint",
    "save_json",
    "save_model_dict",
    "seed_worker",
    "set_global_seed",
    "set_seed",
    "state_dict_to_cpu",
]
