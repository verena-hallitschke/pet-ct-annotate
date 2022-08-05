"""
Module containing metrics that can be calculated on the prediction and ground truth volumes
"""
from typing import Callable
import numpy as np


def dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates the dice score of the given arrays. Assumes 0 to be background and 1 to be foreground

    :param prediction: Binary prediction array, has the same size as ground truth
    :type prediction: np.ndarray
    :param ground_truth: Binary ground truth array, has the same size as the prediction array
    :type ground_truth: np.ndarray
    :return: The dice score of the arrays (0 <= score <= 1.0)
    :rtype: float
    """
    assert prediction.shape == ground_truth.shape, (
        f"Shape mismatch between the prediction and the ground truth: {prediction.shape} != "
        + f"{ground_truth.shape}"
    )

    assert (
        (prediction == 0) | (prediction == 1)
    ).all(), f"Prediction array is not binary!"
    assert (
        (ground_truth == 0) | (ground_truth == 1)
    ).all(), f"Ground truth array is not binary!"

    intersection = np.sum(prediction * ground_truth)

    sum_ped = prediction.sum()
    sum_gt = ground_truth.sum()

    return 2.0 * intersection / (sum_ped + sum_gt)


def get_metric(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Returns the metric matching the given name

    :param metric_name: Name of the metric, case insensitive
    :type metric_name: str
    :raises ValueError: If the name of the metric is unknown
    :return: The function
    :rtype: Callable[[np.ndarray, np.ndarray], float]
    """
    if metric_name.lower() == "dice":
        return dice

    raise ValueError(f"Unknown metric name: {metric_name}!")
