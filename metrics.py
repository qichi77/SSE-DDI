import numpy as np
from sklearn import metrics


METRIC_NAMES = (
    'acc',
    'auc',
    'f1',
    'precision',
    'recall',
    'ap',
)


def do_compute_metrics(
    probas_pred,
    target,
    threshold=0.5,
):
    probabilities = np.asarray(
        probas_pred,
        dtype=np.float64,
    ).reshape(-1)

    labels = np.asarray(
        target,
        dtype=np.int64,
    ).reshape(-1)

    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            'Prediction and target lengths differ: '
            f'{probabilities.shape[0]} != '
            f'{labels.shape[0]}'
        )

    if probabilities.size == 0:
        raise ValueError(
            'Cannot compute metrics on an empty dataset.'
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    acc = metrics.accuracy_score(
        labels,
        predictions,
    )

    if np.unique(labels).size < 2:
        auc = float('nan')
    else:
        auc = metrics.roc_auc_score(
            labels,
            probabilities,
        )

    f1 = metrics.f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    precision = metrics.precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = metrics.recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    if np.sum(labels == 1) == 0:
        ap = float('nan')
    else:
        ap = metrics.average_precision_score(
            labels,
            probabilities,
        )

    return (
        float(acc),
        float(auc),
        float(f1),
        float(precision),
        float(recall),
        float(ap),
    )
