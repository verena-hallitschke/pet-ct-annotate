"""
Module containing utility functions for bounding boxes
"""

from typing import Tuple
import numpy as np

from skimage.measure import regionprops


def get_bounding_box(label: np.ndarray, seed_value: int = 1) -> np.ndarray:
    """
    Finds bounding box in label. Returns the bbox as an array \
    [[start x, start y, start z], [end x, end y, end z]]

    :param label: Array containing the labels where the bounding boxes are extracted
    :type label: np.ndarray
    :param seed_value: Value of the are in the label where the bounding boxes will be extracted, \
        defaults to 1
    :type seed_value: int, optional
    :return: Bounding box matching the label [[start x, start y, start z], [end x, end y, end z]]
    :rtype: np.ndarray
    """
    props = regionprops(label)
    mask_bbox = None
    for box in props:
        if box.label == seed_value:
            mask_bbox = np.array(box.bbox).reshape((2, 3))
            break

    return mask_bbox


def resize_bounding_box(orig_bbox: np.ndarray, multiplier: float) -> np.ndarray:
    """
    Resizes the bounding box using the given multiplier

    :param orig_bbox: Bounding box that will be resized \
        [[start x, start y, start z], [end x, end y, end z]]
    :type orig_bbox: np.ndarray
    :param multiplier: Width and height multiplier
    :type multiplier: float
    :return: Bounding box with new width and height. Values might be < 0!
    :rtype: np.ndarray
    """
    assert multiplier > 0, "Multiplier has to be > 0!"
    widths = orig_bbox[1] - orig_bbox[0]

    new_widths = multiplier * widths

    mid_points = orig_bbox[0] + 0.5 * widths

    new_starts = np.round(mid_points - 0.5 * new_widths).astype(np.int32)
    new_ends = np.round(new_starts + new_widths).astype(np.int32)

    return np.array([new_starts, new_ends], dtype=np.int32)


def move_to_target_width(
    bot: int, top: int, target_width: int, max_val: int, min_val: int = 0
) -> Tuple[int, int]:
    """
    Moves the bounding box values so they are at least target width. In case the target width \
    does not fit into the boundaries [min_val, max_val], the maximum possible width is used.

    Args:
        bot (int): Lower value of the bounding box, >= 0
        top (int): Upper value of the bounding box, >= 0
        target_width (int): Minimum width of the resulting bounding box, > 0
        max_val (int): Maximum possible value of the resulting top value
        min_val (int, optional): Minimum possible value of the resulting bot value. Defaults to 0.

    Returns:
        Tuple[int, int]: New bounding box values (bot, top)
    """

    assert top >= bot, f"Invalid top and bottom values! Bot > top: {bot} > {top}"
    assert top >= 0, f"Invalid top value! Top < 0: {top} < 0"
    assert bot >= 0, f"Invalid bot value! Top < 0: {bot} < 0"

    if top - bot >= target_width and top <= max_val and bot >= min_val:
        return (bot, top)

    current_width = top - bot
    diff = max(0, target_width - current_width)
    half_min = int(round(0.5 * diff))
    moved_b = bot - half_min
    moved_t = top + diff - half_min

    adjuster = max(0, min_val - moved_b) - max(0, moved_t - max_val)

    moved_t = min(max_val, moved_t + adjuster)
    moved_b = max(min_val, moved_b + adjuster)

    return (moved_b, moved_t)


def clip_bounding_box(
    orig_bbox: np.ndarray, max_shape: np.ndarray, min_width: np.ndarray = 10
):
    """
    Clips the given bounding box to [0, max_dim]

    :param orig_bbox: Box that will be clipped. [[start x, start y, start z], [end x, end y, end z]]
    :type orig_bbox: np.ndarray
    :param max_shape: Maximum value for the bounding box. Has to be broadcastable with orig_bbox[1]
    :type max_shape: np.ndarray
    :param min_width: Minimum width of each axis, has to be broadcastable with orig_bbox[0]
    :type min_width: np.ndarray
    """

    orig_bbox[0] = np.maximum(orig_bbox[0], 0)
    orig_bbox[1] = np.minimum(orig_bbox[1], max_shape)

    min_width_arr = np.zeros(orig_bbox.shape[1])
    min_width_arr[:] = min_width

    for ax, width in enumerate(orig_bbox[1] - orig_bbox[0]):
        ax_min_width = min_width_arr[ax]
        if width < ax_min_width:
            orig_bbox[0][ax], orig_bbox[1][ax] = move_to_target_width(
                orig_bbox[0][ax], orig_bbox[1][ax], ax_min_width, max_shape[ax] - 1
            )

    return orig_bbox


def create_image_from_bounding_box(
    orig_bbox: np.ndarray, image_shape: Tuple[int], label_value: int = 1
) -> np.ndarray:
    """
    Creates a np.uint8 image of the given shape where the location of the bounding box is filled \
    with label_value

    :param orig_bbox: Location of the bounding box within the image [[start x, start y, start z], \
        [end x, end y, end z]]
    :type orig_bbox: np.ndarray
    :param image_shape: Target shape of the resulting label
    :type image_shape: Tuple[int]
    :param label_value: Fill value for the bounding box position, defaults to 1
    :type label_value: int, optional
    :return: Array of given shape where the bounding box is marked with label_value
    :rtype: np.ndarray
    """
    assert len(orig_bbox) == 2, "Unknown formatting of bounding box"
    assert len(orig_bbox[0]) == len(image_shape), (
        "Dimensions of bounding box and target shape do not match: "
        + f"{len(orig_bbox[0])} != {len(image_shape)}"
    )
    assert (
        0 <= label_value <= 255
    ), "Unsigned integer overflow, label_value has to be 0 <= label_value <= 255"

    label = np.zeros(image_shape, dtype=np.uint8)
    label[
        orig_bbox[0][0] : orig_bbox[1][0],
        orig_bbox[0][1] : orig_bbox[1][1],
        orig_bbox[0][2] : orig_bbox[1][2],
    ] = label_value

    return label
