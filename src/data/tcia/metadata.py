"""Metadata preprocessing for TCIA Lung-PET-CT-Dx dataset."""
import pandas as pd


def get_filtered_metadata(metadata_path):
    """
    Filter metadata:
        - remove studies without PET images
        - remove series with Secondary Capture Image Storage
        - only use one PT and CT per study
        - choose the CT with the highest amount of images if there are multiple
    """
    metadata = pd.read_csv(metadata_path)
    filtered_metadata = (
        metadata[
            (
                metadata["Study UID"].isin(
                    metadata[(metadata["Modality"] == "PT")]["Study UID"]
                )
            )
            & (
                metadata["SOP Class UID"].isin(
                    ["1.2.840.10008.5.1.4.1.1.128", "1.2.840.10008.5.1.4.1.1.2"]
                )
            )
        ]
        .groupby(["Study UID", "Modality"], as_index=False)
        .apply(lambda group: group.nlargest(1, columns="Number of Images"))
        .groupby("Study UID")
        .filter(lambda x: len(x) == 2)
    )
    # restructure metadata: combine PT and CT modalities for a specific study into one entry
    restructured_metadata = filtered_metadata[
        ["Study UID", "Modality", "File Location", "Subject ID"]
    ]  # [filtered_metadata['Subject ID'].isin(['Lung_Dx-A0255', 'Lung_Dx-A0259', 'Lung_Dx-A0260'])]
    restructured_metadata = (
        restructured_metadata[restructured_metadata["Modality"] == "CT"][
            ["Study UID", "File Location"]
        ]
        .rename(columns={"File Location": "CT File Location"})
        .set_index("Study UID")
        .join(
            restructured_metadata[restructured_metadata["Modality"] == "PT"][
                ["Study UID", "File Location"]
            ]
            .rename(columns={"File Location": "PT File Location"})
            .set_index("Study UID")
        )
    )
    return restructured_metadata
