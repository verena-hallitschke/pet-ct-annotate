"""Helper functions to load datasets in a format suitable for usage in MultimodalDatastore."""
import os
import pandas as pd
from .tcia.metadata import get_filtered_metadata


def load_dataset(
    dataset_type: str, studies_dir: str, split_type: str = "full", **kwargs
):
    """
    Load the dataset given the type and the location.

    :param dataset_type: String describing the type of the dataset. Must be one of
        ["tcia", "autopet", "custom"]
    :param studies_dir: Base directory of the study.
    :param split_type: Split of the dataset to be used. Must be one of
        ["full", "train", "annotate"]
    :param kwargs: Additional paramters given to some dataset loaders.
    """
    if dataset_type not in ["tcia", "autopet", "custom"]:
        raise ValueError(f"Unknown dataset type {dataset_type}.")

    if dataset_type == "tcia":
        annotations_dir = kwargs.get("annotations_dir", None)
        dataset = load_tcia_dataset(studies_dir, annotations_dir)
    elif dataset_type == "autopet":
        dataset = load_autopet_dataset(studies_dir)
    elif dataset_type == "custom":
        dataset = load_custom_dataset(studies_dir)
    return split_dataset(dataset, split_type)


def load_autopet_dataset(metadata_file_dir):
    """
    Load the autoPET dataset.

    :param metadata_file_dir: The directory of the autoPETmeta.csv of the autoPET dataset.
    """
    metadata_file = os.path.join(metadata_file_dir, "autoPETmeta.csv")

    dataset_dataframe = pd.read_csv(metadata_file)
    # filter out patients, using only patients with non-small-cell lung cancer
    dataset_dataframe = dataset_dataframe[
        dataset_dataframe["diagnosis"] == "LUNG_CANCER"
    ]
    for key, filename in zip(
        ["CT", "PET", "Segmentation"], ["CTres.nii.gz", "PET.nii.gz", "SEG.nii.gz"]
    ):
        dataset_dataframe[key] = dataset_dataframe["study_location"].str.slice(start=2)
        dataset_dataframe[key] = dataset_dataframe[key].apply(
            lambda loc, fn=filename: os.path.join(metadata_file_dir, loc, fn)
        )
    dataset_dataframe["id"] = dataset_dataframe["study_location"].str.slice(21, 37)
    # filter out potential missing data
    for key in ["CT", "PET", "Segmentation"]:
        dataset_dataframe = dataset_dataframe[
            dataset_dataframe[key].apply(os.path.isfile)
        ]
    return dataset_dataframe[["id", "CT", "PET", "Segmentation"]]


def load_tcia_dataset(metadata_file_dir, annotations_location=None):
    """
    Load the tcia dataset.

    :param metadata_file_dir: The directory of the metadata.csv of the TCIA dataset.
    :param annotations_location: The base directory of the ground truth annotations.
        If None, do not load annotations.
    """
    metadata_file = os.path.join(metadata_file_dir, "metadata.csv")
    metadata = get_filtered_metadata(metadata_file)
    datasetready_metadata = metadata.rename(
        columns={"Study UID": "id", "CT File Location": "CT", "PT File Location": "PET"}
    )
    # remove the preceding '.' from each relative path and join it with the base path
    for key in ["CT", "PET"]:
        datasetready_metadata[key] = datasetready_metadata[key].str.slice(start=1)
        datasetready_metadata[key] = metadata_file_dir + datasetready_metadata[key]

    # filter out potential missing data points if we're not working on the full dataset
    for key in ["CT", "PET"]:
        datasetready_metadata = datasetready_metadata[
            datasetready_metadata[key].apply(os.path.exists)
        ]

    if annotations_location is not None and os.path.isdir(annotations_location):
        # annotations are found by removing the metadata directory from the CT location and
        # appending the remainder to the base annotation directory
        datasetready_metadata["Segmentation"] = (
            datasetready_metadata["CT"]
            .str.slice(len(metadata_file_dir) + 1)
            .apply(
                lambda x: os.path.join(annotations_location, x, "segmentation.nii.gz")
            )
        )
        # once again filter missing data
        datasetready_metadata = datasetready_metadata[
            datasetready_metadata["Segmentation"].apply(os.path.exists)
        ]

    # recover Study UID to rename it
    datasetready_metadata = datasetready_metadata.reset_index()
    datasetready_metadata = datasetready_metadata.rename(columns={"Study UID": "id"})
    return datasetready_metadata


def load_custom_dataset(metadata_file_dir):
    """
    Directly load a dataset from a custom metadata.csv file.
    The file must contain the columns "id", "CT", "PET", and optionally "Segmentation".

    :param metadata_file_dir: The directory of the custom metadata.csv.
    """
    # pylint doesn't correctly infer the type of pd.read_csv
    # pylint: disable=unsubscriptable-object
    dataset_dataframe: pd.DataFrame = pd.read_csv(
        os.path.join(metadata_file_dir, "metadata.csv")
    )
    if not all(key in dataset_dataframe for key in ["id", "CT", "PET"]):
        raise ValueError("Custom metadata.csv is missing required keys.")

    # filter out missing data
    keys_with_paths = ["CT", "PET"]
    if "Segmentation" in dataset_dataframe:
        keys_with_paths += ["Segmentation"]
    for key in keys_with_paths:
        dataset_dataframe = dataset_dataframe[
            dataset_dataframe[key].apply(os.path.exists)
        ]

    return dataset_dataframe


def split_dataset(dataset: pd.DataFrame, split_type: str, split_ratio=0.9):
    """
    Split the dataset, return the split indicated by the split type.

    :param dataset: The dataset to split.
    :param split_type: The type of split to return. Must be one of ["full", "train", "annotate"]
    :param split_ratio: The ratio of data used for the training split.
    """
    if split_type not in ["full", "train", "annotate"]:
        raise ValueError(f"Unknown split_type {split_type}")
    if split_type == "full":
        return dataset
    split_index = round(split_ratio * len(dataset)) + 1
    if split_type == "train":
        return dataset.iloc[:split_index]
    if split_type == "annotate":
        return dataset.iloc[split_index:]
    return None
