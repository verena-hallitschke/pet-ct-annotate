"""Make data training-ready."""
import argparse
import data.tcia.preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess for annotation files of the TCIA Lung-PET-CT-Dx dataset."
    )
    parser.add_argument(
        "metadata_file", help="Location of the metadata.csv file of the dataset."
    )
    parser.add_argument(
        "annotations_location", help="Base directory of the unpacked annotation files."
    )
    parser.add_argument(
        "target_location",
        help="Target base location of the preprocessed annotation files.",
    )
    arguments = parser.parse_args()
    data.tcia.preprocess.preprocess_annotations(
        arguments.metadata_file,
        arguments.annotations_location,
        arguments.target_location,
    )
