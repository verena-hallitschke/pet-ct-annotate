"""Transforms used for saving data."""
from typing import Any, Dict, Hashable
import os
import time
import multiprocessing
import logging
import monai.transforms.transform
import monai.data.image_writer

logger = logging.getLogger(__name__)


def save_image_with_writer(
    writer: monai.data.image_writer.ImageWriter, path: os.PathLike
):
    """
    Uses the given writer to write an image to the given path.

    :param writer: The MONAI ImageWriter to use.
    :param path: The path to save the image to.
    """
    writer.write(path)


class SaveImageToFiled(monai.transforms.transform.Transform):
    # pylint: disable=too-few-public-methods
    """Transform that saves a single image of the data dictionary to a file.

    :param key: Key of the image in the dictionary that will be saved. The image is expected to
        have a single channel and be in [spatial x spatial x spatial] format, unless a channel
        dimension is specified by channel_dim.
    :param target_dir: Target directory to save the image to.
    :param file_prefix: Prefix to prepend to the filename.
    :param prefix_key: Additional prefix to be read from the data dictionary from this key.
    :param meta_dict_key: Optional key of the meta dict to associate with the image.
    :param async_save: True to save in a separate process, False to save in the current process.
    :param channel_dim: The dimension of the image that indicates channels.
    """

    def __init__(
        self,
        key: str,
        target_dir: str,
        file_prefix: str = "image_",
        prefix_key: str = None,
        meta_dict_key: str = None,
        async_save: bool = True,
        channel_dim=None,
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.key = key
        self.target_dir = target_dir
        self.prefix = file_prefix
        self.prefix_key = prefix_key
        self.meta_dict_key = meta_dict_key
        self.async_save = async_save
        self.channel_dim = channel_dim

    def __call__(self, d: Dict[Hashable, Any]):
        try:
            os.makedirs(self.target_dir, exist_ok=True)
            writer = monai.data.image_writer.ITKWriter()

            writer.set_data_array(d[self.key], channel_dim=self.channel_dim)
            if self.meta_dict_key is not None:
                writer.set_metadata(d[self.meta_dict_key])

            prefix = self.prefix
            if self.prefix_key and type(d[self.prefix_key]) == str:
                prefix += d[self.prefix_key]
            filename = (
                prefix + time.strftime("_%Y-%m-%d-%H%M%S", time.localtime()) + ".nii.gz"
            )

            target_path = os.path.join(self.target_dir, filename)
            if self.async_save:
                write_process = multiprocessing.Process(
                    target=save_image_with_writer, args=(writer, target_path)
                )
                write_process.start()
            else:
                writer.write(os.path.join(self.target_dir, filename))

        except TypeError:
            logger.info("Could not convert image for saving.")
        return d
