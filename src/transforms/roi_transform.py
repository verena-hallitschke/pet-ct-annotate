"""
Wrapper to perform a given transformation on a Region of Interest (ROI)
"""
import sys
import os
import logging
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np
from monailabel.scribbles.transforms import InteractiveSegmentationTransform
from torch import Tensor
import torch

# Add root folder to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# pylint: disable=wrong-import-position
# Need to add to path beforehand
from util.bbox import clip_bounding_box, get_bounding_box, resize_bounding_box


# pylint: disable=too-few-public-methods
# Monai interface
class RegionOfInterestTransform(InteractiveSegmentationTransform):
    """
        Wrapper for transformation(s) to perform them on a ROI

        Workflow:

            1. Calculates bounding box using the given labels in data[cropping_mask_key] using \
                seed_value
            2. Resizes bounding box by factor roi_factor
            3. Crops all images with the keys given in cropping_keys (assumes 3d channels first)
            4. Performs transformation(s) on data with the cropped images
            5. Takes keys from post_transform_map and patches the matching images in the \
                transformation result back to the original size. The entries outside of the ROI \
                are filled using the matching value in post_transform_map.
            6. Outputs are added to the dictionary and returned

        :param transformations: Transformation or list of transformations that should be called
        :type transformations: Callable or List[Callable]

        :param cropping_keys: Keys of the numpy arrays in data that will be resized. It is assumed \
            that they all have the same size and have the channel first
        :type cropping_keys: List[str]

        :param cropping_mask_key: Key of the labels used for calculating the bounding box
        :type cropping_mask_key: str

        :param post_transform_map: Maps the output keys to a fill value or a function. Needed when \
            filling the output outside the ROI. In case it is a function, it is called using the \
            output of the transformation(s) as input.

            Example:

                * post_transform_map = { "output": np.max } -> Use max value in ROI as fill value
                * post_transform_map = { "output": 1 } -> Use 1 as fill value

        :type post_transform_map: Dict[str, Union[Callable, float, int]]

        :param seed_value: Value that marks the segmentation in the label, defaults to 1
        :type seed_value: int, optional

        :param roi_factor: Factor by which the bounding box in the label will be scaled, \
            defaults to 2.0
        :type roi_factor: float, optional

        :param handle_missing_bbox: Defines behavior in case the bounding box (basis for ROI) \
            cannot be computed. There are the following options or this setting:

            * skip: (default) Calculation is skipped
            * constant: Creating a constant array using post_transform_map
            * full: Full images will be given to the transformation
            * raise: Raises a ValueError
        :param name: Name of this class during logging. Defaults to "ROI <transformation name>"
        :type handle_missing_bbox: str
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        transformations: Union[Callable, List[Callable]],
        cropping_keys: List[str],
        cropping_mask_key: str,
        post_transform_map: Dict[
            str, Union[Callable[[np.ndarray], Union[float, int]], float, int]
        ],
        seed_value: int = 1,
        roi_factor: float = 2.0,
        handle_missing_bbox: str = "full",
        name: str = None,
    ):
        super().__init__()
        self.transformations = transformations
        if not isinstance(transformations, Iterable):
            self.transformations = [self.transformations]

        self.cropping_keys = cropping_keys
        self.cropping_mask_key = cropping_mask_key

        self.seed_value = seed_value
        self.roi_factor = roi_factor
        self.post_transform_map = post_transform_map

        self.handle_missing_bbox = handle_missing_bbox.lower()
        if self.handle_missing_bbox not in ["skip", "full", "constant", "raise"]:
            logging.warning(
                "Unknown setting for 'handle_missing_bbox': %s! Setting to 'skip'!",
                self.handle_missing_bbox,
            )
            self.handle_missing_bbox = "skip"

        if name is not None:
            self.__class__.__name__ = name

        elif len(self.transformations) == 1:
            self.__class__.__name__ = (
                f"ROI {self.transformations[0].__class__.__name__}"
            )
        else:
            self.__class__.__name__ = (
                f"ROI {self.transformations[0].__class__.__name__} +"
            )

    def __call__(self, data: Any):

        mask_arr = data.get(self.cropping_mask_key)

        if mask_arr is not None:
            mask = np.array(mask_arr, copy=True).squeeze().astype(np.uint8)
            mask_box = get_bounding_box(mask, seed_value=self.seed_value)
        else:
            mask_box = None

        if mask_box is None:
            logging.warning(f"No bounding box found in '{self.cropping_mask_key}'!")

            if self.handle_missing_bbox == "full":
                # run on full image
                logging.warning("Processing full image!")
                result_dict = data
                for trafo in self.transformations:
                    result_dict = trafo(result_dict)
                return result_dict

            if self.handle_missing_bbox == "constant":
                # Set outputs to constant value
                logging.warning("Returning constant values!")
                for key, fill_value_or_func in self.post_transform_map.items():
                    final_fill_val = fill_value_or_func

                    if callable(fill_value_or_func):
                        final_fill_val = fill_value_or_func(data[key])

                    data[key] = np.full(
                        data[self.cropping_mask_key].shape, fill_value=final_fill_val,
                    )

                return data

            if self.handle_missing_bbox == "raise":
                raise ValueError("Could not calculate bounding box!")

            # Skip
            logging.warning("Skipping transformation!")
            return data

        mask_box = clip_bounding_box(
            resize_bounding_box(mask_box, self.roi_factor), mask.shape
        )

        cropped_dict = data.copy()

        # Crop
        for key in self.cropping_keys:
            current_image = data[key]

            if isinstance(current_image, Tensor):
                current_image = torch.clone(current_image)
            else:
                current_image = current_image.copy()

            cropped_dict[key] = current_image[
                ...,
                mask_box[0][0] : mask_box[1][0],
                mask_box[0][1] : mask_box[1][1],
                mask_box[0][2] : mask_box[1][2],
            ]

        result_dict = cropped_dict
        for trafo in self.transformations:
            result_dict = trafo(result_dict)

        # Patch back
        for key, fill_value_or_func in self.post_transform_map.items():
            final_fill_val = fill_value_or_func
            if callable(fill_value_or_func):
                final_fill_val = fill_value_or_func(result_dict[key])

            patched_image = np.full(
                data[self.cropping_mask_key].shape,
                fill_value=final_fill_val,
                dtype=result_dict[key].dtype,
            )
            patched_image[
                :,
                mask_box[0][0] : mask_box[1][0],
                mask_box[0][1] : mask_box[1][1],
                mask_box[0][2] : mask_box[1][2],
            ] = result_dict[key]

            data[key] = patched_image

        return data
