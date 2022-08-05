"""A MONAI Datastore for multimodal data."""
import io
import pathlib
from typing import Any, Dict, List
import os
import shutil
import logging
import pandas as pd
import monailabel.datastore.utils.convert
import monailabel.interfaces.datastore
import monailabel.utils.others.generic

logger = logging.getLogger(__name__)


def to_bytes(uri):
    """
    Read the file given by the uri and return it as bytes.
    """
    return io.BytesIO(pathlib.Path(uri).read_bytes())


class MultimodalDatastore(monailabel.interfaces.datastore.Datastore):
    # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """A Datastore for local images with multiple modalities."""

    MODALITY_DELIMITER = "#-#-#"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_keys: List,
        id_key="ID",
        label_path=".",
        label_key="label_" + monailabel.interfaces.datastore.DefaultLabelTag.FINAL,
        info_key="info",
        file_extension=".nii.gz",
        convert_to_nifti=True,
    ) -> None:
        # pylint: disable=too-many-arguments
        """
        Creates a new MultimodalDatastore object.

        :param dataframe: The pandas dataframe this datastore uses as info storage.
        :param image_keys: The keys / column names of the dataframe that contain
            the different modalities for a single datapoint.
        :param id_key: The key / column name of the dataframe that is used to identify datapoints.
        :param label_path: Base path where labels will be saved to.
        :param label_key: The key / column name of the labels.
        :param info_key: Column name where additional file info is stored.
        :param file_extension: Default file extension to use for new label files.
        """
        super().__init__()
        self._name = "Multimodal Datastore"
        self._desc = "A datastore for multimodal data."
        self._dataframe = dataframe
        self._image_keys = image_keys
        self._id_key = id_key
        self._label_path = label_path
        self._label_key = label_key
        self._info_key = info_key
        self._file_extension = file_extension
        self._convert_to_nifti = convert_to_nifti

        if self._label_key not in self._dataframe:
            self._dataframe[self._label_key] = ""
            self._init_labels()
        if self._info_key not in self._dataframe:
            self._dataframe[self._info_key] = None  # lazy initialization

    def _init_labels(self):
        ids_with_labels = map(
            lambda filename: os.path.basename(filename)[
                : -len(self._file_extension)
            ],  # trim path
            filter(
                os.path.isfile,  # only load actual files
                map(
                    lambda filename: os.path.join(
                        self._label_path, filename
                    ),  # prepend path
                    os.listdir(self._label_path),
                ),
            ),
        )
        for id_key in ids_with_labels:
            if id_key in self._dataframe[self._id_key].values:
                self._dataframe.loc[
                    self._dataframe[self._id_key] == id_key, self._label_key
                ] = os.path.join(self._label_path, id_key + self._file_extension)

    def name(self) -> str:
        """
        Return the human-readable name of the datastore

        :return: the name of the dataset
        """
        return self._name

    def set_name(self, name: str):
        """
        Set the name of the datastore

        :param name: a human-readable name for the datastore
        """
        self._name = name

    def description(self) -> str:
        """
        Return the user-set description of the dataset

        :return: the user-set description of the dataset
        """
        return self._desc

    def set_description(self, description: str):
        """
        A human-readable description of the datastore

        :param description: string for description
        """
        self._desc = description

    def datalist(self) -> List[Dict[str, Any]]:
        """
        Return a dictionary of image and label pairs corresponding to the 'image' and 'label'
        keys respectively

        :return: the {'image': image, 'label': label} pairs for training
        """
        return (
            self._dataframe.loc[
                self._dataframe[self._label_key] != "",
                [self._id_key, self._label_key] + self._image_keys,
            ]
            .rename(columns={self._id_key: "image", self._label_key: "label"})
            .to_dict("records")
        )

    def get_labels_by_image_id(self, image_id: str) -> Dict[str, str]:
        """
        Retrieve all label ids for the given image id

        :param image_id: the desired image's id
        :return: label ids mapped to the appropriate `LabelTag` as Dict[LabelTag, str]
        """
        return (
            {
                monailabel.interfaces.datastore.DefaultLabelTag.FINAL: self._dataframe[
                    self._dataframe[self._id_key] == image_id
                ][self._label_key].iat[0]
            }
            if any(self._dataframe[self._id_key] == image_id)
            else None
        )

    def get_label_by_image_id(self, image_id: str, tag: str) -> str:
        """
        Retrieve label id for the given image id and tag

        :param image_id: the desired image's id
        :param tag: matching tag name
        :return: label id
        """
        return self.get_labels_by_image_id(image_id).get(tag, "")

    def get_image(self, image_id: str, _=None) -> Any:
        """
        Retrieve image object based on image id

        :param image_id: the desired image's id
        :param params: any optional params
        :return: return the "image"
        """
        # This is actually never called in MONAILabel, just return an empty dict for now.
        return {}

    def get_image_uri(self, image_id: str) -> str:
        """
        Retrieve image uri based on image id

        :param image_id: the desired image's id
        :return: return the image uri
        """
        image_id, modality_key = image_id.split(self.MODALITY_DELIMITER)
        if modality_key in self._image_keys:
            return self.get_image_modality_uri(image_id, modality_key)
        return ""

    def get_image_modality_uri(
        self, image_id: str, modality_key: str, convert=None
    ) -> str:
        """
        Retrieve the uri of one modality of one image id

        :param image_id: the desired image's id
        :param modality_key: the desired modality's key
        :param convert: whether to convert a directory to a single image
        :return: return the uri of the modality
        """
        convert = self._convert_to_nifti if convert is None else convert
        uri = self._dataframe[self._dataframe[self._id_key] == image_id][
            modality_key
        ].iloc[0]
        if os.path.isdir(uri) and convert:
            # got a dir, treat as dicom dir and convert to nifti
            return self._get_cached_or_convert(image_id, modality_key, uri)
        return uri

    def _get_cached_or_convert(self, image_id: str, modality_key: str, uri: str):
        """
        Return the cached converted image if it exists, otherwise convert the dir and cache it.
        """
        if not os.path.isdir(uri):
            raise ValueError(f"Can't convert non-directory {uri}.")
        # Make sure the dataframe has the necessary columns
        cached_key = modality_key + "_cachedfile"
        if cached_key not in self._dataframe:
            self._dataframe[cached_key] = ""

        cached_file = self._dataframe[self._dataframe[self._id_key] == image_id][
            cached_key
        ].iloc[0]
        if cached_file == "" or not os.path.isfile(cached_file):
            logger.info(
                "No cached file found for %s-%s, converting.", image_id, modality_key
            )
            # no cached file, convert and save it
            cached_file = monailabel.datastore.utils.convert.dicom_to_nifti(uri)
            self._dataframe.loc[
                self._dataframe[self._id_key] == image_id, cached_key
            ] = cached_file
        return cached_file

    def get_label(self, label_id: str, label_tag: str, _=None) -> Any:
        """
        Retrieve image object based on label id

        :param label_id: the desired label's id
        :param label_tag: the matching label's tag
        :param params: any optional params
        :return: return the "label"
        """
        uri = self.get_label_uri(label_id, label_tag)
        return to_bytes(uri) if uri else None

    def get_label_uri(self, label_id: str, label_tag: str) -> str:
        """
        Retrieve label uri based on image id

        :param label_id: the desired label's id
        :param label_tag: the matching label's tag
        :return: return the label uri
        """
        return self._dataframe[self._dataframe[self._id_key] == label_id][
            self._label_key
        ].iloc[0]

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        """
        Get the image information for the given image id

        :param image_id: the desired image id
        :return: image info as a list of dictionaries Dict[str, Any]
        """
        info = (
            self._dataframe[self._dataframe[self._id_key] == image_id].iloc[0][
                self._info_key
            ]
            if any(self._dataframe[self._id_key] == image_id)
            else {}
        )
        info = info if info is not None else {}
        info.update(
            {
                "image_ids": [
                    image_id + self.MODALITY_DELIMITER + modality_key
                    for modality_key in self._image_keys
                ],
                "paths": [
                    self.get_image_modality_uri(image_id, modality_key, convert=False)
                    for modality_key in self._image_keys
                ],
                "modalities": self._image_keys,
            }
        )
        return info if info is not None else {}

    def get_label_info(self, label_id: str, label_tag: str) -> Dict[str, Any]:
        """
        Get the label information for the given label id

        :param label_id: the desired label id
        :param label_tag: the matching label tag
        :return: label info as a list of dictionaries Dict[str, Any]
        """
        return {}

    def get_labeled_images(self) -> List[str]:
        """
        Get all images that have a corresponding final label

        :return: list of image ids List[str]
        """
        return self._dataframe[self._dataframe[self._label_key] != ""][
            self._id_key
        ].to_list()

    def get_unlabeled_images(self) -> List[str]:
        """
        Get all images that have no corresponding final label

        :return: list of image ids List[str]
        """
        return self._dataframe[self._dataframe[self._label_key] == ""][
            self._id_key
        ].to_list()

    def list_images(self) -> List[str]:
        """
        Return list of image ids available in the datastore

        :return: list of image ids List[str]
        """
        return self._dataframe[self._id_key].to_list()

    def refresh(self) -> None:
        """
        Refresh the datastore
        """
        raise NotImplementedError

    def add_image(
        self, image_id: str, image_filename: str, image_info: Dict[str, Any]
    ) -> str:
        """
        Save a image for the given image id and return the newly saved image's id

        :param image_id: the image id for the image;  If None then base filename will be used
        :param image_filename: the path to the image file
        :param image_info: additional info for the image
        :return: the image id for the saved image filename
        """
        raise NotImplementedError

    def remove_image(self, image_id: str) -> None:
        """
        Remove image for the datastore.  This will also remove all associated labels.

        :param image_id: the image id for the image to be removed from datastore
        """
        raise NotImplementedError

    def save_label(
        self,
        image_id: str,
        label_filename: str,
        label_tag: str,
        label_info: Dict[str, Any],
    ) -> str:
        """
        Save a label for the given image id and return the newly saved label's id

        :param image_id: the image id for the label
        :param label_filename: the path to the label file
        :param label_tag: the user-provided tag for the label
        :param label_info: additional info for the label
        :return: the label id for the given label filename
        """
        destination_file = os.path.join(
            self._label_path, image_id + self._file_extension
        )
        shutil.copy(label_filename, destination_file)
        self._dataframe.loc[
            self._dataframe[self._id_key] == image_id, self._label_key
        ] = destination_file
        return image_id

    def remove_label(self, label_id: str, label_tag: str) -> None:
        """
        Remove label from the datastore

        :param label_id: the label id for the label to be removed from datastore
        :param label_tag: the label tag for the label to be removed from datastore
        """
        raise NotImplementedError

    def update_image_info(self, image_id: str, info: Dict[str, Any]) -> None:
        """
        Update (or create a new) info tag for the desired image

        :param image_id: the id of the image we want to add/update info
        :param info: a dictionary of custom image information Dict[str, Any]
        """
        self.get_image_info(image_id).update(info)

    def update_label_info(
        self, label_id: str, label_tag: str, info: Dict[str, Any]
    ) -> None:
        """
        Update (or create a new) info tag for the desired label

        :param label_id: the id of the label we want to add/update info
        :param label_tag: the matching label tag
        :param info: a dictionary of custom label information Dict[str, Any]
        """
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        """
        Return current statistics of datastore
        """
        return {
            "total": len(self.list_images()),
            "completed": len(self.get_labeled_images()),
        }

    def json(self):
        """
        Return json representation of datastore
        """
        return self.datalist()
