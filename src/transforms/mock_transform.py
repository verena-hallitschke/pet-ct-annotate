"""
Contains a transformation that mocks the the user input
"""
import os
import sys

from typing import Callable
import numpy as np
from monailabel.scribbles.transforms import InteractiveSegmentationTransform
from monai.data.utils import affine_to_spacing

# Add root folder to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# pylint: disable=wrong-import-position
# Need to use sys path extension in order to ensure correct importing
from util.mock_user_input import (
    create_ellipsoid_from_label,
    create_scribbles_image_walking,
    create_seed_from_label,
)
from util.bbox import (
    get_bounding_box,
    resize_bounding_box,
    clip_bounding_box,
    create_image_from_bounding_box,
)

# pylint: disable=too-few-public-methods
# This is due to the monai interface
class CreateMockMask(InteractiveSegmentationTransform):
    """
    Mocks the user input by randomly sampling from the given label.

    :param label_mask_key: Key of the label in the dictionary, defaults to "label"
    :type label_mask_key: str, optional
    :param image_key: Key of the image in the dictionary. Needed for spacing, defaults to "image"
    :type image_key: str, optional
    :param seed_type: Type of the resulting seed. Can be one of "ellipsoid", "seed" and "scribble",
        defaults to "ellipsoid"
    :type seed_type: str, optional
    :param output_key: Key where the resulting mask will be saved to, defaults to "mask"
    :type output_key: str, optional
    :param seed_value: Value of the cells that are used for sampling, defaults to 1
    :type seed_value: int, optional
    :param result_fill_value: Value of the selected cells in the resulting mask, defaults to 1
    :type result_fill_value: int, optional
    :param mock_func_kwargs: Additional input arguments that can be given to the user input
        mocking function, i.e. the spacing for ellipsoid creation
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        label_mask_key: str = "label",
        image_key: str = "image",
        seed_type: str = "ellipsoid",
        output_key: str = "mask",
        seed_value: int = 1,
        result_fill_value: int = 1,
        **mock_func_kwargs,
    ):

        super().__init__("meta_dict")
        self.label_mask_key = label_mask_key

        self.mocking_func: Callable = create_ellipsoid_from_label
        self.spacing_meta_key = f"{image_key}_meta_dict"

        if seed_type.lower() == "seed":
            self.mocking_func = create_seed_from_label
            self.spacing_meta_key = None
        elif seed_type.lower() == "scribble":
            self.mocking_func = create_scribbles_image_walking
            self.spacing_meta_key = None

        self.seed_value = seed_value
        self.result_fill_value = result_fill_value
        self.output_key = output_key
        self.mock_func_kwargs = mock_func_kwargs

    def __call__(self, data):
        data_dict = dict(data)
        label = np.asarray(self._fetch_data(data, self.label_mask_key)).squeeze()

        if self.spacing_meta_key is not None:
            if "spacing" in data_dict[self.spacing_meta_key]:
                self.mock_func_kwargs["spacing"] = list(
                    reversed(data_dict[self.spacing_meta_key]["spacing"])
                )
            elif "affine" in data_dict[self.spacing_meta_key]:
                self.mock_func_kwargs["spacing"] = list(
                    affine_to_spacing(data_dict[self.spacing_meta_key]["affine"])
                )

        gen_mask = self.mocking_func(
            label,
            seed_value=self.seed_value,
            result_fill_value=self.result_fill_value,
            **self.mock_func_kwargs,
        )

        data_dict[self.output_key] = np.array([gen_mask])

        return data_dict


class CreateBackgroundMockMask(InteractiveSegmentationTransform):
    """
    Mocks the user input by randomly sampling from the background around given label.

    :param label_mask_key: Key of the label in the dictionary, defaults to "label"
    :type label_mask_key: str, optional
    :param image_key: Key of the image in the dictionary. Needed for spacing, defaults to "image"
    :type image_key: str, optional
    :param output_key: Key where the resulting mask will be saved to, defaults to "mask"
    :type output_key: str, optional
    :param seed_value: Value of the cells that are used for sampling, defaults to 1
    :type seed_value: int, optional
    :param result_fill_value: Value of the selected cells in the resulting mask, defaults to 1
    :type result_fill_value: int, optional
    :param mock_func_kwargs: Additional input arguments that can be given to the user input
        mocking function, i.e. the spacing for ellipsoid creation
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        label_mask_key: str = "label",
        image_key: str = "image",
        output_key: str = "mask",
        seed_value: int = 1,
        result_fill_value: int = 1,
        **mock_func_kwargs,
    ):

        super().__init__("meta_dict")
        self.label_mask_key = label_mask_key

        self.seed_value = seed_value
        self.result_fill_value = result_fill_value
        self.output_key = output_key
        self.mock_func_kwargs = mock_func_kwargs
        self.spacing_meta_key = f"{image_key}_meta_dict"

    def __call__(self, data):
        data_dict = dict(data)
        label = np.asarray(self._fetch_data(data, self.label_mask_key)).squeeze()

        # find correct mask_bbox with specified seed value
        mask_bbox = get_bounding_box(label.astype(int), self.seed_value)
        if mask_bbox is None:
            # no foreground pixels use full image
            mask_bbox = np.ndarray([[0, 0, 0], label.shape], dtype=np.int32)
        else:
            # Calculate ROI around label
            mask_bbox = clip_bounding_box(
                resize_bounding_box(mask_bbox, 3.0), label.shape
            )

        inverted_label = create_image_from_bounding_box(
            mask_bbox, label.shape, label_value=1
        )

        # Clip with original label
        inverted_label = np.where(label == self.seed_value, 0, inverted_label)

        # Get spacing
        spacing = [1.0, 1.0, 1.0]
        if "spacing" in data_dict[self.spacing_meta_key]:
            spacing = list(reversed(data_dict[self.spacing_meta_key]["spacing"]))
        elif "affine" in data_dict[self.spacing_meta_key]:
            spacing = list(
                affine_to_spacing(data_dict[self.spacing_meta_key]["affine"])
            )

        # Calculate ellipsoid
        gen_mask = create_ellipsoid_from_label(
            inverted_label,
            seed_value=1,
            result_fill_value=self.result_fill_value,
            spacing=spacing,
            **self.mock_func_kwargs,
        )

        # Clip with original label
        gen_mask = np.where(label == self.seed_value, 0, gen_mask)

        data_dict[self.output_key] = np.array([gen_mask])

        return data_dict
