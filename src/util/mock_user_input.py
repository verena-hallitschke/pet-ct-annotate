"""
This module is used to mock the user scribbles during training
"""

# pylint: disable=invalid-sequence-index
# Pylint does not work well with arbitrary indexes
from typing import List, Tuple
import numpy as np
from skimage.draw import ellipsoid, line_aa

from util.bbox import get_bounding_box


def get_single_seed_from_label(label: np.ndarray, seed_value: int = 1):
    """
    Samples a random position from the 3 dimensional array label.

    :param label: 3 dimensional array of the size of the input image.
        1 means foreground, 0 background
    :param seed_value: Value of the cells in the label that represent the seed

    :return: Numpy array of shape len(label.shape) containing a random coordinate as seed
    """

    bbox_voxels = np.argwhere(label == seed_value)

    random_position = bbox_voxels[np.random.randint(0, len(bbox_voxels))]

    return random_position


def get_scribble_from_label(
    label: np.ndarray, seed_value: int = 1
) -> Tuple[np.ndarray, int]:
    """
    Samples scribble points from a label mask along two randomly selected axes.

    :param label: 3D-array of type uint8. Cells with value seed_value are
        interpreted as cells of interest
    :type label: np.ndarray
    :param seed_value: Value of the cells that are labeled, defaults to 1
    :type seed_value: int, optional
    :return: Tuple: Array containing the selected points, axis that was dropped during
        calculation (points have constant value)
    :rtype: Tuple[np.ndarray, int]
    """
    # pylint: disable=too-many-locals
    # Sample point in label
    pos = get_single_seed_from_label(label, seed_value=seed_value)

    # Sample axis
    axis_identity = [0, 1, 2]
    drop_axis = np.random.randint(0, 3)
    axis_identity.remove(drop_axis)
    walking_axis: int = int(np.random.randint(0, 2))
    walking_axis_identity = axis_identity[walking_axis]
    other_axis_identity = axis_identity[1 - walking_axis]

    sliced_ind = [slice(None), slice(None), slice(None)]
    sliced_ind[drop_axis] = pos[drop_axis]
    image_slice = np.where(label[tuple(sliced_ind)] == seed_value, 1, 0)

    label_bbox = get_bounding_box(image_slice, 1)
    resolution = 4.0
    bbox_width = label_bbox[3] - label_bbox[1]
    bbox_height = label_bbox[2] - label_bbox[0]
    max_points = int(np.ceil(min(bbox_width, bbox_height) / resolution))

    # Get indices of other axis that contain label voxels

    step_sum = np.sum(image_slice, axis=walking_axis)
    step_argwhere = np.argwhere(step_sum > 0)

    num_sampled_points = min(np.random.randint(2, max_points), len(step_argwhere))

    selected_index_arr_ax_1 = np.random.choice(
        step_argwhere.squeeze(), size=num_sampled_points
    )

    # Null unselected rows/cols
    current_slice = [slice(None), slice(None)]
    scribble_points = np.empty((len(selected_index_arr_ax_1), 3), dtype=np.uint32)

    for index, guide in enumerate(selected_index_arr_ax_1):
        # walk through walking axis, select points
        current_slice[1 - walking_axis] = guide
        sampled_index = np.random.choice(
            np.argwhere(image_slice[tuple(current_slice)] == 1).squeeze()
        )
        delete_slice = [guide, guide]
        delete_slice[walking_axis] = sampled_index
        # set sampled position to 0 to avoid it being sampled again
        image_slice[tuple(delete_slice)] = 0

        # set point
        scribble_points[index][walking_axis_identity] = sampled_index
        scribble_points[index][other_axis_identity] = guide
        scribble_points[index][drop_axis] = pos[drop_axis]

    return np.sort(scribble_points), drop_axis


def get_random_ellipsoid(
    position: np.array,
    image_shape: Tuple[int],
    spacing: List[float] = None,
    result_fill_value: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Creates a mask describing a random ellipsoid

    :param position: Position in 3D space where ellipsoid is generated
    :type position: np.array
    :param image_shape: Shape of the resulting mask
    :type image_shape: Tuple[int]
    :param spacing: Spacing of the volume image, defaults to [1.0, 1.0, 1.0]
    :type spacing: List[float], optional
    :param result_fill_value: Value of the cells in the mask that will be part of the ellipsoid,
        defaults to 1
    :param axes_scale Scaling factor used to determine maximum size of the ellipsoid. \
        Max_size = axes_scale * image_shape
    :type result_fill_value: int, optional
    :return: Mask that has the shape image_shape and random dimensions
    :rtype: np.ndarray
    """
    if spacing is None:
        spacing = [1.0, 1.0, 1.0]
    # Sample axes lengths
    scales = kwargs.get("axes_scale", np.array([0.1, 0.1, 0.1]))  # x, y, z
    min_width = 2
    axes = np.maximum(
        np.round(
            np.random.rand(3)
            * (
                np.min(
                    [
                        0.5 * scales * (image_shape - position),
                        np.maximum(position, min_width),
                    ],
                    axis=0,
                )
                - min_width
            )
            + min_width
        ),
        np.array([min_width] * 3),
    )

    ellip_base = ellipsoid(axes[0], axes[1], axes[2], spacing=spacing)

    # ensure starts are within the image
    starts = np.maximum(position, axes.astype(np.uint8)) - axes.astype(np.uint8)
    # ensure ends will be within the image while also covering the whole ellipsoid shape
    starts = np.minimum(
        starts, np.array(image_shape, np.uint) - np.array(ellip_base.shape, np.uint)
    ).astype(int)

    ends = starts + ellip_base.shape
    ends = ends.astype(int)

    mask = np.zeros(image_shape)

    # Match middle of ellip_base with pos
    mask[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]] = np.where(
        ellip_base, result_fill_value, 0
    )

    return mask.astype(np.uint8)


def create_seed_from_label(
    label: np.ndarray, seed_value: int = 1, result_fill_value: int = 1
) -> np.ndarray:
    """
    Randomly selects a seed from label and creates a mask that contains it.

    :param label: Label that bases the seed creation. Cells of interest have the value
        seed_value, uint8
    :type label: np.ndarray
    :param seed_value: Value of the cells that are used for sampling, defaults to 1
    :type seed_value: int, optional
    :param result_fill_value: Value of the selected cells in the resulting mask, defaults to 1
    :type result_fill_value: int, optional
    :return: Mask of the shape label.shape where the selected cells have the value \
        result_fill_value, uint8
    :rtype: np.ndarray
    """
    mask = np.zeros(label.shape, dtype=np.uint8)
    seed = get_single_seed_from_label(label, seed_value=seed_value)

    mask[tuple(seed)] = result_fill_value

    return mask


def create_scribbles_image_walking(
    label: np.ndarray,
    seed_value: int = 1,
    result_fill_value: int = 1,
    scribble_depth: int = 2,
) -> np.ndarray:
    """
    Creates a scribble mask by dropping one dimension

    :param label: Label that bases the seed creation. Cells of interest have the value
        seed_value, uint8
    :type label: np.ndarray
    :param seed_value: Value of the cells that are used for sampling, defaults to 1
    :type seed_value: int, optional
    :param result_fill_value: Value of the selected cells in the resulting mask, defaults to 1
    :type result_fill_value: int, optional
    :param scribble_depth: Determines how many layers contain the scribble.
        In order to emulate the dropped dimension, the resulting scribble can be repeated in the
        layers around the selected one, defaults to 2
    :type scribble_depth: int, optional
    :return: Mask of the shape label.shape where the scribble cells have the value \
        result_fill_value, uint8
    :rtype: np.ndarray
    """
    # pylint: disable=too-many-locals
    assert scribble_depth > 0 and np.all(
        np.array(label.shape) > scribble_depth
    ), f"Invalid scribble_depth: {scribble_depth}"
    # Sample scribble
    scribble_points, dropped_axis = get_scribble_from_label(
        label, seed_value=seed_value
    )

    axis_val = scribble_points[0][dropped_axis]

    scribble_mask = np.zeros(label.shape, dtype=np.uint8)

    scribble_slice = [slice(None), slice(None), slice(None)]
    scribble_slice[dropped_axis] = axis_val

    current_mask = scribble_mask[tuple(scribble_slice)]

    previous_point: List[int] = []
    for current_point in scribble_points:
        # Do rasterization in 2D
        current_2d = list(current_point.copy())
        del current_2d[dropped_axis]  # remove entry

        # Draw lines
        if len(previous_point) > 0:
            row_slice, col_slice, _ = line_aa(*previous_point, *current_2d)
            current_mask[row_slice, col_slice] = result_fill_value

        previous_point = current_2d

    scribble_slice[dropped_axis] = axis_val - int(round(0.5 * scribble_depth))

    for _ in range(scribble_depth):
        # expand in depth

        if scribble_slice[dropped_axis] == axis_val:
            # Skip original layer
            continue
        scribble_mask[tuple(scribble_slice)] = current_mask
        scribble_slice[dropped_axis] += 1
    return scribble_mask


def create_ellipsoid_from_label(
    label: np.ndarray,
    spacing: List[float] = None,
    seed_value: int = 1,
    result_fill_value: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Creates a mask containing a ellipsoid based on the given label

    :param label: Label that bases the seed creation. Cells of interest have the value
        seed_value, uint8
    :type label: np.ndarray
    :param spacing: Spacing of the volume image, defaults to [1.0, 1.0, 1.0]
    :type spacing: List[float], optional
    :param seed_value: Value of the cells that are used for sampling, defaults to 1
    :type seed_value: int, optional
    :param result_fill_value: Value of the selected cells in the resulting mask, defaults to 1
    :type result_fill_value: int, optional
    :param axes_scale Scaling factor used to determine maximum size of the ellipsoid. \
        Max_size = axes_scale * image_shape
    :return: Mask of the shape label.shape where the ellipsoid cells have the value \
        result_fill_value, uint8
    :rtype: np.ndarray
    """
    if spacing is None:
        spacing = [1.0, 1.0, 1.0]
    # Sample ellipsoid position
    pos = get_single_seed_from_label(label, seed_value=seed_value)

    return get_random_ellipsoid(
        pos, label.shape, spacing=spacing, result_fill_value=result_fill_value, **kwargs
    )
