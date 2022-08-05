"""Transforms used for ensuring consistency with the user annotation."""
from typing import Any, Dict, Hashable
import monai.transforms.transform
import numpy as np


class EnsurePrediction(monai.transforms.transform.Transform):
    # pylint: disable=too-few-public-methods
    """Ensure that foreground and background annotations are respected in the prediction.

    :param pred_key: Key of the prediction that will be changed.
    :param input_key: Input of the user containing foreground and background annotations.
    :param foreground_label: Label of the foreground in the input.
    :param background_label: Label of the background in the input.
    """

    def __init__(
        self,
        pred_key: str,
        input_key: str,
        foreground_label: int = 3,
        background_label: int = 2,
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.pred_key = pred_key
        self.input_key = input_key
        self.foreground_label = foreground_label
        self.background_label = background_label

    def __call__(self, d: Dict[Hashable, Any]):
        d[self.pred_key] = np.where(
            d[self.input_key].squeeze() == self.foreground_label, 1.0, d[self.pred_key]
        )
        d[self.pred_key] = np.where(
            d[self.input_key].squeeze() == self.background_label, 0.0, d[self.pred_key]
        )
        return d
