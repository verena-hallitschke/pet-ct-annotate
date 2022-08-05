"""Utility for preprocessing the TCIA Lung-PET-CT-Dx dataset."""
import os
import re
import xml.etree.ElementTree
import skimage.draw
import itk
import pydicom
import numpy as np
from .metadata import get_filtered_metadata


def get_bounding_boxes(path):
    """
    Load bounding boxes from the PASCAL VOC xml file at path.
    """
    xmltree = xml.etree.ElementTree.parse(path)
    bounding_boxes = []
    for bndbox in xmltree.getroot().iter("bndbox"):
        xmin = int(bndbox.find("xmin").text)
        xmax = int(bndbox.find("xmax").text)
        ymin = int(bndbox.find("ymin").text)
        ymax = int(bndbox.find("ymax").text)
        bounding_boxes.append((xmin, ymin, xmax, ymax))
    return bounding_boxes


def draw_bounding_boxes(array, bounding_boxes, bounding_box_value=1):
    """
    Draw filled rectangles inside the array.
    """
    for bbox in bounding_boxes:
        rect_idx1, rect_idx2 = skimage.draw.rectangle(
            (bbox[0], bbox[1]), (bbox[2], bbox[3])
        )
        array[rect_idx1, rect_idx2] = bounding_box_value


def filename_to_sliceindex(filename):
    """
    Returns the extracted slice index from a filename.
    """
    return int(filename[filename.find("-") + 1 : filename.find(".")]) - 1


def save_segmentation(segmentation, path):
    """
    Save the segmentation array with correctly ordered axes.
    """
    itk.imwrite(
        itk.image_view_from_array(segmentation.transpose(2, 1, 0)[::-1, :, :]), path
    )


def load_annotations(path):
    """
    Returns all annotation filenames found at path and subdirectories.
    """
    all_annotations = []
    for root, _, files in os.walk(path):
        for filename in files:
            all_annotations.append(os.path.join(root, filename))
    return all_annotations


def preprocess_study(study_root, annotations_paths, annotations_sopuids):
    """
    Returns the combined 3D segmentation for a study.
    """
    slice_files = [
        file for file in os.listdir(study_root) if re.match(r"\d-\d*\.dcm", file)
    ]
    segmentation = None
    for filename in slice_files:
        dcm = pydicom.dcmread(os.path.join(study_root, filename))
        annotation_slice = np.zeros_like(dcm.pixel_array)
        annotation_filename = dcm.SOPInstanceUID + ".xml"
        if annotation_filename in annotations_sopuids:
            annotation_path = os.path.join(
                annotations_paths[annotations_sopuids.index(annotation_filename)],
                annotation_filename,
            )
            draw_bounding_boxes(annotation_slice, get_bounding_boxes(annotation_path))
        segmentation = (
            np.zeros(
                (
                    annotation_slice.shape[0],
                    annotation_slice.shape[1],
                    len(slice_files),
                )
            )
            if segmentation is None
            else segmentation
        )  # lazy initialization, as we have all the required dimension sizes at this point
        segmentation[:, :, filename_to_sliceindex(filename)] = annotation_slice
    return segmentation


def preprocess_annotations(
    metadatafile_location, annotations_location, target_location
):
    """
    Convert the TCIA Lung-PET-CT-Dx dataset annotations to 3D volumetric images
    and saves them at the target location.
    """
    metadata_root = os.path.dirname(metadatafile_location)
    restructured_metadata = get_filtered_metadata(metadatafile_location)

    annotations = load_annotations(annotations_location)
    annotations_paths = list(map(os.path.dirname, annotations))
    annotations_sopuids = list(map(os.path.basename, annotations))

    for _, study in restructured_metadata.iterrows():
        study_location = study["CT File Location"][2:]
        study_root = os.path.join(metadata_root, study_location)
        if not os.path.isdir(study_root):
            continue
        segmentation = preprocess_study(
            study_root, annotations_paths, annotations_sopuids
        )
        if segmentation.max() > 0:
            # only save a segmentation file if we actually have non-zero data
            os.makedirs(
                os.path.join(target_location, study_location), exist_ok=True,
            )
            save_segmentation(
                segmentation,
                os.path.join(target_location, study_location, "segmentation.nii.gz"),
            )
