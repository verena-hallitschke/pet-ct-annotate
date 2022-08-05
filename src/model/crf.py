"""
Module containing a wrapper for SimpleCRF
"""

from typing import List

# pylint: disable=c-extension-no-member
# C module, not detected by pylint
import denseCRF3D
import numpy as np


def array_grey_transform(arr: np.ndarray) -> np.ndarray:
    """
    Clips an array to the interval [0, 255]

    :param arr: Numpy array that will be processed
    :type arr: np.ndarray
    :return: Array where the values are in the domain [0, 255]
    :rtype: uint8 np.ndarray
    """
    normalized_arr = arr.copy()

    normalized_arr -= arr.min()
    normalized_arr /= normalized_arr.max()
    normalized_arr *= 255

    normalized_arr = normalized_arr.round()

    normalized_arr[normalized_arr < 0] = 0
    normalized_arr[normalized_arr > 255] = 255

    return normalized_arr.astype(np.uint8)


class MultimodalSimpleCRF:
    """
    Multimodal 3D Conditional Random Field (CRF) segmentation. Wraps the SimpleCRF interface.

    :param crf_parameters: Dictionary containing settings for the CRF inference.
        See SimpleCRF for more
    :type crf_parameters: Dict[str, Any]
    :param number_of_modalities: Number of modalities used for inference
    :type number_of_modalities: int
    """

    def __init__(self, number_of_modalities: int = 2, **kwargs):
        """
        Initializes settings of CRF. The keyword arguments are checked for the
        following crf parameters:
        * MaxIterations
        * PosW
        * PosRStd
        * PosCStd
        * PosZStd
        * BilateralW
        * BilateralRStd
        * BilateralCStd
        * BilateralZStd
        * ModalityNum
        * BilateralModsStds

        If given the default setting of these parameters is overwritten with the input value.
        See SimpleCRF for further information on these parameters.

        :param number_of_modalities: Number of modalities that will be processed, defaults to 2
        :type number_of_modalities: int, optional
        """
        self.crf_parameters = {}
        self.crf_parameters["MaxIterations"] = kwargs.get("MaxIterations", 2.0)
        self.crf_parameters["PosW"] = kwargs.get("PosW", 2.0)
        self.crf_parameters["PosRStd"] = kwargs.get("PosRStd", 5)
        self.crf_parameters["PosCStd"] = kwargs.get("PosCStd", 5)
        self.crf_parameters["PosZStd"] = kwargs.get("PosZStd", 5)
        self.crf_parameters["BilateralW"] = kwargs.get("BilateralW", 3.0)
        self.crf_parameters["BilateralRStd"] = kwargs.get("BilateralRStd", 5.0)
        self.crf_parameters["BilateralCStd"] = kwargs.get("BilateralCStd", 5.0)
        self.crf_parameters["BilateralZStd"] = kwargs.get("BilateralZStd", 5.0)
        self.crf_parameters["ModalityNum"] = kwargs.get("ModalityNum", 1)
        self.crf_parameters["BilateralModsStds"] = kwargs.get(
            "BilateralModsStds", (5.0 for _ in range(number_of_modalities))
        )

        self.number_of_modalities = number_of_modalities

    def predict(
        self,
        images: List[np.ndarray],
        foreground_probability_map: np.ndarray,
        background_probability_map: np.ndarray,
    ) -> np.ndarray:
        """
        Performs a segmentation based of the given modalities (images) and the foreground and
        background probability maps. All arrays have to have the same shape

        :param images: List of modalities, has to match self.number_of_modalities.
            Each of the images has to be a 3D array in order [depth, height, width]
        :type images: List[np.ndarray]
        :param foreground_probability_map: Numpy array with values between 0.0 and 1.0 defining the
            probabilities of the foreground class at every voxel in the image
        :type foreground_probability_map: np.ndarray
        :param background_probability_map: Numpy array with values between 0.0 and 1.0 defining the
            probabilities of the background class at every voxel in the image
        :type background_probability_map: np.ndarray
        :return: Binary array containing the resulting segmentation. 0 = no object, 1 = object.
            Has the same shape as the modalities
        :rtype: np.ndaray
        """
        assert len(images) > 0, "Missing input images"
        assert (
            foreground_probability_map.min() >= 0.0
            and background_probability_map.min() >= 0.0
            and foreground_probability_map.max() <= 1.0
            and background_probability_map.max() <= 1.0
        ), "The values of the probability maps have to be in the domain [0, 1]"

        assert (
            len(images) == self.number_of_modalities
        ), "Number of images does not match number of modalities: {} != {}".format(
            len(images), self.number_of_modalities
        )

        input_image_shape = images[0]

        for current in images:
            assert (
                current.shape == input_image_shape
            ), "All arrays have to be of the same shape: {} != {}".format(
                current.shape, input_image_shape
            )

        assert (
            background_probability_map.shape == input_image_shape
        ), "All arrays have to be of the same shape: {} != {}".format(
            background_probability_map.shape, input_image_shape
        )
        assert (
            foreground_probability_map.shape == input_image_shape
        ), "All arrays have to be of the same shape: {} != {}".format(
            foreground_probability_map.shape, input_image_shape
        )

        # clip image to interval 0, 255
        normalized_images = np.array(images, copy=False)
        normalized_images = array_grey_transform(normalized_images)

        # Convert image to [D, H, W, C] # TODO error in SimpleCRF docu? C++ uses (WHDC)
        normalized_images = normalized_images.transpose([1, 2, 3, 0])

        # Transform probability map to tensor with [D, H, W, number of classes]
        stacked_probability_map = np.array(
            [foreground_probability_map, background_probability_map], copy=False
        ).transpose([1, 2, 3, 0])

        labels = denseCRF3D(
            normalized_images, stacked_probability_map, self.crf_parameters
        )

        # TODO necessary post processing
        return labels
