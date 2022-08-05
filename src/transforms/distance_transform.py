"""
Contains classes for geodesic distance transformation in monai
"""

from typing import Any, Optional

# pylint: disable=c-extension-no-member
import GeodisTK
from monailabel.scribbles.transforms import InteractiveSegmentationTransform
import monai.data.utils
import numpy as np

# pylint: disable=too-few-public-methods
# This is due to the monai interface
class SeedsToMask(InteractiveSegmentationTransform):
    """
    Creates a volume mask using the given seeds. The volume has the value
    "seed_value" where a seed was places and the value "fill_value" everywhere else.
    The volume has the same dimensions as the image.

    Args:
        image_key: The key containing the image in the dictionary
        input_label_key: Key that references the list of seeds
        output_key: Key where the mask will be stored.
            If not given defaults to <input_label_key>_mask
        seed_value: Value used to mark seeds in mask
    """

    def __init__(
        self,
        image_key: str = "image",
        input_label_key: str = "background",
        output_key: Optional[str] = None,
        seed_value: int = 1,
    ):
        super().__init__("meta_dict")

        self.image_key = image_key
        self.input_label_key = input_label_key
        self.ouput_key = (
            output_key if output_key is not None else f"{input_label_key}_mask"
        )
        self.fill_value = 0
        self.seed_value = seed_value

    def __call__(self, data: Any) -> dict:

        data_dict = dict(data)
        image = np.asarray(self._fetch_data(data, self.image_key)).squeeze()

        seed_list = self._fetch_data(data, self.input_label_key)

        mask = np.full(image.shape, self.fill_value, dtype=np.uint8)

        for entry in seed_list:
            mask[tuple(entry)] = self.seed_value

        data_dict[self.ouput_key] = np.array([mask])
        return data_dict


# pylint: disable=too-few-public-methods
# This is due to the monai interface
class GeodesicUserInputTransform(InteractiveSegmentationTransform):
    """
    Performs a geodesic transform on the image and adds it to the dictionary.
    Requires the image and a mask of the same shape to be in the dictionary

    Args:
        image_key: The key containing the image in the dictionary, float32
        scribbles_mask_key: The key containing the mask in the dictionary, uint8
        output_key: Key where the mask will be stored.
        num_iterations: Numbers of iterations the raster scan is performed
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image_key: str = "image",
        scribbles_mask_key: str = "label",
        output_key: str = "geo_transformed_image",
        seed_value: int = 1,
        num_iterations: int = 3,
    ) -> None:
        super().__init__("meta_dict")
        self.image_key = image_key
        self.scribbles_mask_key = scribbles_mask_key
        self.output_key = output_key
        self.num_iterations = num_iterations
        self.transform_lambda = 1.0
        self.seed_value = seed_value

    def __call__(self, data: Any) -> dict:
        data_dict = dict(data)

        # get label position
        image = self._fetch_data(data, self.image_key)

        mask = np.asarray(
            self._fetch_data(data, self.scribbles_mask_key), np.uint8
        ).squeeze()

        meta_dict_key = self.image_key + "_meta_dict"
        if meta_dict_key not in data_dict:
            raise ValueError(f"Meta dict not found for image {self.image_key}.")
        if "spacing" in data_dict[meta_dict_key]:
            spacing = list(
                reversed(data_dict[meta_dict_key]["spacing"])
            )  # self._fetch_data(data, self.spacing_key)
        elif "affine" in data_dict[meta_dict_key]:
            spacing = list(
                monai.data.utils.affine_to_spacing(data_dict[meta_dict_key]["affine"])
            )
        else:
            raise ValueError(f"Could not determine spacing for {self.image_key}.")

        image = np.asarray(image, np.float32).squeeze()

        assert mask.shape == image.shape

        mask = np.where(mask == self.seed_value, 1, 0).astype(np.uint8)

        geo_dis_image = GeodisTK.geodesic3d_raster_scan(
            image, mask, spacing, self.transform_lambda, self.num_iterations
        )

        data_dict[self.output_key] = np.array([geo_dis_image])

        return data_dict
