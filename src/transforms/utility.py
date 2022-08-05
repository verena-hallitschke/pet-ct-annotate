"""Utility transforms for processing outputs to the required input of other transforms."""
from typing import Any, Dict, Hashable, List, Type
import numpy as np
import monailabel.scribbles.transforms
import monai.transforms.transform
import torch


class AffineCopy(monailabel.scribbles.transforms.InteractiveSegmentationTransform):
    # pylint: disable=too-few-public-methods
    """Copy over the affine metadata from the source image to the target image key."""

    def __init__(
        self,
        source_key_prefix: str = "image",
        target_key_prefix: str = "pred",
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(meta_key_postfix)
        self.source_key_prefix = source_key_prefix
        self.target_key_prefix = target_key_prefix

    def __call__(self, d: Dict[Hashable, Any]):
        return self._copy_affine(d, self.source_key_prefix, self.target_key_prefix)


class ConvertBinaryToTwoChannels(monai.transforms.transform.Transform):
    # pylint: disable=too-few-public-methods
    """Convert Numpy arrays with the shape [HWD] to Numpy arrays with the shape [2HWD],
    where the first channel is (1-p) and the second channel is p for every p in the original array.
    """

    def __init__(self, keys: List):
        self.keys = keys

    def __call__(self, d: Dict[Hashable, Any]):
        for key in self.keys:
            squeeze = len(d[key].shape) != 3
            d[key] = np.stack([1.0 - d[key], d[key]])
            if squeeze:
                d[key] = d[key].squeeze()
        return d


# pylint: disable=too-few-public-methods
# Monai interface
class FreeCUDACache(monai.transforms.transform.Transform):
    """
    Utility transformation that simply frees the CUDA cache if available
    """

    def __call__(self, data):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return data


class FallBackTransform(monai.transforms.transform.Transform):
    """
        Performs transformations in the given lists until the current transformation succeeds.

        Listens to the errors defined in 'listening_errors' in order to determine whether the \
        execution was successful.

        Raises raised_error_class in case none of the transformations succeed.

        Example Call:
            FallBackTransform([trafo_a, trafo_b], listening_errors=[ValueError], \
                raised_error_class=TypeError)

        Case 1:
            trafo_a succeeds -> return result

        Case 2:
            trafo_a raises ValueError -> try trafo_b -> trafo_b raises ValueError -> tried last \
                transformation -> raise raised_error_class (TypeError)

        Case 3:
            trafo_a raises ValueError -> try trafo_b -> trafo_b raises RuntimeError (not in \
                listening_errors) -> immediately raise RuntimeError

        Case 4:
            trafo_a raises ValueError -> try trafo_b -> trafo_b succeeds -> return result

        :param transformations: List of transformation in the order they should be applied
        :type transformations: List[Type[monai.transforms.transform.Transform]]
        :param listening_errors: Errors that will trigger the execution of the next fallback \
            option, defaults to [ValueError, TypeError]
        :type listening_errors: List[Type[Exception]], optional
        :param raised_error_class: Error that will be raised in case none of the transformations \
            were successful, defaults to ValueError
        :type raised_error_class: Type[Exception], optional
    """

    def __init__(
        self,
        transformations: List[Type[monai.transforms.transform.Transform]],
        listening_errors: List[Type[Exception]] = None,
        raised_error_class: Type[Exception] = ValueError,
    ) -> None:
        assert len(transformations) != 0, "Missing 'transformations' parameter!"
        self.transformations = transformations

        self.listening_errors = listening_errors
        if self.listening_errors is None:
            self.listening_errors = [ValueError, TypeError]

        self.raised_error_class = raised_error_class

    def __call__(self, data: Any) -> Any:
        last_error = None
        error_trafo_name = None
        for current_trafo in self.transformations:
            try:
                result = current_trafo(data)
            # pylint: disable=catching-non-exception
            # Needed so user can select exceptions
            except tuple(self.listening_errors) as e:
                last_error = e
                error_trafo_name = current_trafo.__class__.__name__
                continue

            return result

        if last_error is not None:
            raise self.raised_error_class(
                "Tried all fallback transformations! Last error: "
                + f"{last_error.__class__.__name__} while executing {error_trafo_name}!"
            ) from last_error

        raise self.raised_error_class("Tried all fallback transformations!")


class RandomRepeatTransform(monai.transforms.transform.Transform):
    """
    Repeat the given transform a random number of times, combining their output by taking
        the maximum of each output.

    :param transform: The transform to repeat.
    :param output_key: The key that the transform will write to.
    :param repeat_times_min: The minimum number of times to repeat the transform.
    :param repeat_times_max: The maximum number of times to repeat the transform.
    """

    def __init__(
        self,
        transform: monai.transforms.transform.Transform,
        output_key: str,
        repeat_times_min: int,
        repeat_times_max: int,
    ):
        super().__init__()
        self.transform = transform
        self.output_key = output_key
        self.repeat_times_min = repeat_times_min
        self.repeat_times_max = repeat_times_max

    def __call__(self, data: Any) -> Any:
        times = np.random.randint(self.repeat_times_min, self.repeat_times_max + 1)
        transform_output = data.get(self.output_key, None)
        for _ in range(times):
            transform_data = self.transform(data)
            if transform_output is not None:
                transform_output = np.maximum(
                    transform_output, transform_data[self.output_key]
                )
            else:
                transform_output = transform_data[self.output_key]
        data[self.output_key] = transform_output
        return data
