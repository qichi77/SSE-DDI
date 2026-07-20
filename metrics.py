from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES: Tuple[str, ...] = (
    "acc",
    "auc",
    "f1",
    "precision",
    "recall",
    "ap",
)

DEFAULT_CLASSIFICATION_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class BinaryMetricResult:
    """Metrics and confusion counts for one evaluation split."""

    acc: float
    auc: float
    f1: float
    precision: float
    recall: float
    ap: float
    threshold: float
    num_instances: int
    num_positive: int
    num_negative: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "acc": float(self.acc),
            "auc": float(self.auc),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "ap": float(self.ap),
            "threshold": float(self.threshold),
            "num_instances": int(self.num_instances),
            "num_positive": int(self.num_positive),
            "num_negative": int(self.num_negative),
            "true_positive": int(self.true_positive),
            "true_negative": int(self.true_negative),
            "false_positive": int(self.false_positive),
            "false_negative": int(self.false_negative),
        }

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return ``(ACC, AUC, F1, precision, recall, AP)``."""

        return (
            float(self.acc),
            float(self.auc),
            float(self.f1),
            float(self.precision),
            float(self.recall),
            float(self.ap),
        )


def _as_float_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")

    return array


def _as_binary_vector(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)

    if raw.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if raw.dtype.kind in {"f", "c"} and not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} contains NaN or infinity")

    try:
        array = raw.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} cannot be converted to integer binary labels"
        ) from exc

    unique_values = np.unique(array)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"{name} must contain only 0 and 1; observed "
            f"{unique_values.tolist()}"
        )

    return array


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    return threshold


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            "probabilities must lie in [0, 1]. "
            "Use compute_binary_metrics_from_logits() for logits."
        )


def sigmoid(logits: Any) -> np.ndarray:
    """Numerically stable NumPy sigmoid."""

    values = _as_float_vector(logits, "logits")
    probabilities = np.empty_like(values)

    nonnegative = values >= 0
    probabilities[nonnegative] = 1.0 / (
        1.0 + np.exp(-values[nonnegative])
    )

    negative = ~nonnegative
    exponentials = np.exp(values[negative])
    probabilities[negative] = exponentials / (1.0 + exponentials)

    return probabilities


def probabilities_to_predictions(
    probabilities: Any,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> np.ndarray:
    """Convert probabilities to zero/one predictions."""

    threshold = _validate_threshold(threshold)
    probability_array = _as_float_vector(
        probabilities,
        "probabilities",
    )
    _validate_probabilities(probability_array)

    return (
        probability_array >= threshold
    ).astype(np.int64)


def compute_binary_metric_result(
    probabilities: Any,
    labels: Any,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> BinaryMetricResult:
    """Compute all manuscript-reported metrics for one split.

    AUC is returned as NaN when the split contains only one class. AP is
    returned as NaN when the split contains no positive samples.
    """

    threshold = _validate_threshold(threshold)

    probability_array = _as_float_vector(
        probabilities,
        "probabilities",
    )
    label_array = _as_binary_vector(
        labels,
        "labels",
    )

    if probability_array.shape != label_array.shape:
        raise ValueError(
            "probabilities and labels must have the same flattened shape; "
            f"observed {probability_array.shape} and {label_array.shape}"
        )

    _validate_probabilities(probability_array)

    prediction_array = (
        probability_array >= threshold
    ).astype(np.int64)

    auc = (
        float("nan")
        if np.unique(label_array).size < 2
        else float(roc_auc_score(label_array, probability_array))
    )

    ap = (
        float("nan")
        if int(np.sum(label_array == 1)) == 0
        else float(
            average_precision_score(
                label_array,
                probability_array,
            )
        )
    )

    matrix = confusion_matrix(
        label_array,
        prediction_array,
        labels=[0, 1],
    )
    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    return BinaryMetricResult(
        acc=float(
            accuracy_score(
                label_array,
                prediction_array,
            )
        ),
        auc=auc,
        f1=float(
            f1_score(
                label_array,
                prediction_array,
                zero_division=0,
            )
        ),
        precision=float(
            precision_score(
                label_array,
                prediction_array,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                label_array,
                prediction_array,
                zero_division=0,
            )
        ),
        ap=ap,
        threshold=threshold,
        num_instances=int(label_array.size),
        num_positive=int(np.sum(label_array == 1)),
        num_negative=int(np.sum(label_array == 0)),
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def compute_binary_metrics(
    probabilities: Any,
    labels: Any,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """Return metrics and counts as a dictionary."""

    return compute_binary_metric_result(
        probabilities=probabilities,
        labels=labels,
        threshold=threshold,
    ).as_dict()


def compute_binary_metrics_from_logits(
    logits: Any,
    labels: Any,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """Compute metrics directly from model logits."""

    return compute_binary_metrics(
        probabilities=sigmoid(logits),
        labels=labels,
        threshold=threshold,
    )


def do_compute_metrics(
    probas_pred: Any,
    target: Any,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Tuple[float, float, float, float, float, float]:
    """Backward-compatible API used by the original repository."""

    return compute_binary_metric_result(
        probabilities=probas_pred,
        labels=target,
        threshold=threshold,
    ).as_tuple()


def metric_dict_to_tuple(
    metrics: Mapping[str, Any],
) -> Tuple[float, float, float, float, float, float]:
    """Return a metric mapping in manuscript column order."""

    missing = [
        metric_name
        for metric_name in METRIC_NAMES
        if metric_name not in metrics
    ]
    if missing:
        raise KeyError(f"Missing metric keys: {missing}")

    return (
        float(metrics["acc"]),
        float(metrics["auc"]),
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["ap"]),
    )


def _extract_metric(
    run_result: Mapping[str, Any],
    metric_name: str,
    scenario: Optional[str],
) -> float:
    source: Mapping[str, Any]

    if scenario is None:
        source = run_result
    else:
        if scenario not in run_result:
            raise KeyError(f"Run result is missing scenario {scenario!r}")

        nested = run_result[scenario]
        if not isinstance(nested, Mapping):
            raise TypeError(
                f"Scenario {scenario!r} must contain a metric mapping"
            )
        source = nested

    if metric_name not in source:
        raise KeyError(f"Run result is missing metric {metric_name!r}")

    value = source[metric_name]
    return float("nan") if value is None else float(value)


def aggregate_repeated_runs(
    run_results: Sequence[Mapping[str, Any]],
    scenario: Optional[str] = None,
    metric_names: Sequence[str] = METRIC_NAMES,
    ddof: int = 1,
    ignore_nan: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate repeated runs as mean ± standard deviation.

    ``ddof=1`` computes sample standard deviation. The function also records
    individual values and percentage-form means/deviations for direct export
    to manuscript tables.
    """

    if not run_results:
        raise ValueError("run_results cannot be empty")
    if not isinstance(ddof, (int, np.integer)) or int(ddof) < 0:
        raise ValueError("ddof must be a non-negative integer")

    ddof = int(ddof)
    metric_names = tuple(str(name) for name in metric_names)

    if not metric_names:
        raise ValueError("metric_names cannot be empty")

    output: Dict[str, Dict[str, Any]] = {}

    for metric_name in metric_names:
        values = np.asarray(
            [
                _extract_metric(
                    run_result,
                    metric_name,
                    scenario,
                )
                for run_result in run_results
            ],
            dtype=np.float64,
        )

        finite_values = values[np.isfinite(values)]

        if ignore_nan:
            working_values = finite_values
        else:
            working_values = values

        if working_values.size == 0:
            mean = float("nan")
            std = float("nan")
        elif not ignore_nan and not np.all(np.isfinite(working_values)):
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(np.mean(working_values))
            std = (
                float("nan")
                if working_values.size <= ddof
                else float(
                    np.std(
                        working_values,
                        ddof=ddof,
                    )
                )
            )

        output[metric_name] = {
            "mean": mean,
            "std": std,
            "mean_percent": (
                100.0 * mean
                if math.isfinite(mean)
                else float("nan")
            ),
            "std_percent": (
                100.0 * std
                if math.isfinite(std)
                else float("nan")
            ),
            "values": values.tolist(),
            "num_runs": int(values.size),
            "num_finite_runs": int(finite_values.size),
            "ddof": ddof,
        }

    return output


def aggregate_scenarios(
    run_results: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    metric_names: Sequence[str] = METRIC_NAMES,
    ddof: int = 1,
    ignore_nan: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate test, S1, S2, or other named scenarios."""

    if not scenarios:
        raise ValueError("scenarios cannot be empty")

    return {
        str(scenario): aggregate_repeated_runs(
            run_results=run_results,
            scenario=str(scenario),
            metric_names=metric_names,
            ddof=ddof,
            ignore_nan=ignore_nan,
        )
        for scenario in scenarios
    }


def format_mean_std(
    mean: float,
    std: float,
    percent: bool = True,
    decimals: int = 2,
    unavailable: str = "NA",
) -> str:
    """Format one result as ``mean ± std``."""

    mean = float(mean)
    std = float(std)

    if not math.isfinite(mean) or not math.isfinite(std):
        return unavailable

    multiplier = 100.0 if percent else 1.0
    return (
        f"{multiplier * mean:.{int(decimals)}f} "
        f"± {multiplier * std:.{int(decimals)}f}"
    )


def format_aggregate_table(
    aggregate: Mapping[str, Mapping[str, Any]],
    percent: bool = True,
    decimals: int = 2,
) -> Dict[str, str]:
    """Format all manuscript metrics for a result table."""

    formatted: Dict[str, str] = {}

    for metric_name in METRIC_NAMES:
        if metric_name not in aggregate:
            continue

        summary = aggregate[metric_name]
        formatted[metric_name] = format_mean_std(
            mean=float(summary["mean"]),
            std=float(summary["std"]),
            percent=percent,
            decimals=decimals,
        )

    return formatted


# ---------------------------------------------------------------------------
# Safe compatibility helpers for code that imported the old manual functions.
# ---------------------------------------------------------------------------

def positive(labels: Any) -> int:
    return int(np.sum(_as_binary_vector(labels, "labels") == 1))


def negative(labels: Any) -> int:
    return int(np.sum(_as_binary_vector(labels, "labels") == 0))


def _paired_binary_vectors(
    labels: Any,
    predictions: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    label_array = _as_binary_vector(labels, "labels")
    prediction_array = _as_binary_vector(
        predictions,
        "predictions",
    )

    if label_array.shape != prediction_array.shape:
        raise ValueError("labels and predictions must have the same shape")

    return label_array, prediction_array


def true_positive(labels: Any, predictions: Any) -> int:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return int(
        np.sum(
            (label_array == 1)
            & (prediction_array == 1)
        )
    )


def false_positive(labels: Any, predictions: Any) -> int:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return int(
        np.sum(
            (label_array == 0)
            & (prediction_array == 1)
        )
    )


def true_negative(labels: Any, predictions: Any) -> int:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return int(
        np.sum(
            (label_array == 0)
            & (prediction_array == 0)
        )
    )


def false_negative(labels: Any, predictions: Any) -> int:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return int(
        np.sum(
            (label_array == 1)
            & (prediction_array == 0)
        )
    )


def binary_precision(labels: Any, predictions: Any) -> float:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return float(
        precision_score(
            label_array,
            prediction_array,
            zero_division=0,
        )
    )


def binary_recall(labels: Any, predictions: Any) -> float:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return float(
        recall_score(
            label_array,
            prediction_array,
            zero_division=0,
        )
    )


def binary_f1(labels: Any, predictions: Any) -> float:
    label_array, prediction_array = _paired_binary_vectors(
        labels,
        predictions,
    )
    return float(
        f1_score(
            label_array,
            prediction_array,
            zero_division=0,
        )
    )


__all__ = [
    "METRIC_NAMES",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "BinaryMetricResult",
    "sigmoid",
    "probabilities_to_predictions",
    "compute_binary_metric_result",
    "compute_binary_metrics",
    "compute_binary_metrics_from_logits",
    "do_compute_metrics",
    "metric_dict_to_tuple",
    "aggregate_repeated_runs",
    "aggregate_scenarios",
    "format_mean_std",
    "format_aggregate_table",
    "positive",
    "negative",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "binary_precision",
    "binary_recall",
    "binary_f1",
]
