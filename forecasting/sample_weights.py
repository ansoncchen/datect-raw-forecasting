"""
Sample weight utilities for spike-focused regression/classification.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_sample_weights_for_classification(
    y_binary: Iterable[int],
    positive_weight: float,
    negative_weight: float,
) -> np.ndarray:
    """
    Build sample weights for binary labels.
    """
    labels = np.asarray(list(y_binary), dtype=int)
    weights = np.full(labels.shape, float(negative_weight), dtype=float)
    weights[labels == 1] = float(positive_weight)
    return weights


def compute_spike_focused_weights(
    da_values: Iterable[float],
    spike_threshold: float,
    spike_weight: float,
    non_spike_weight: float,
) -> np.ndarray:
    """
    Compute regression sample weights by upweighting spike events.
    """
    values = np.asarray(list(da_values), dtype=float)
    is_spike = values > float(spike_threshold)
    return compute_sample_weights_for_classification(
        is_spike.astype(int),
        positive_weight=spike_weight,
        negative_weight=non_spike_weight,
    )
