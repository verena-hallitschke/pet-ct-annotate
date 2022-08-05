"""Model tasks for the MONAILabel server."""
import os
from typing import Any, Callable, Sequence
import numpy as np

import torch.optim
import torch.nn
import monai.inferers
import monai.transforms
import monailabel.tasks.train.basic_train
import monailabel.interfaces.tasks.infer
from monailabel.scribbles.transforms import (
    ApplyGraphCutOptimisationd,
    MakeLikelihoodFromScribblesGMMd,
    MakeISegUnaryd,
)

from transforms.utility import ConvertBinaryToTwoChannels, FallBackTransform
from transforms.roi_transform import RegionOfInterestTransform
from transforms.logging_transform import SaveImageToFiled
from transforms.ensure_annotation_transform import EnsurePrediction


def init_transforms_load_and_normalize():
    """
    Initialize the transforms relating to loading and normalizing the data.
    """
    return [
        monai.transforms.LoadImaged(["CT", "PET", "label"]),
        monai.transforms.EnsureTyped(["CT", "PET", "label"]),
        monai.transforms.EnsureChannelFirstd(keys=["CT", "PET", "label"]),
        monai.transforms.ResampleToMatchd("PET", "CT_meta_dict"),
        monai.transforms.ScaleIntensityRanged(
            keys=["CT"], a_min=-1024, a_max=2 ** 12 - 1, b_min=0, b_max=1
        ),
        monai.transforms.ScaleIntensityRanged(
            keys=["PET"], a_min=0, a_max=2 ** 16 - 1, b_min=0, b_max=1
        ),
    ]


class DummyInferer(monai.inferers.Inferer):
    def __call__(self, *args: Any, **kwargs: Any):
        return None


class GraphCutTask(monailabel.interfaces.tasks.infer.InferTask):
    """MONAILabel entry point for inference."""

    def __init__(self, use_roi_post_processing: bool = True,) -> None:
        """Create the Infertask that only uses GraphCut.

        :param use_roi_post_processing: Whether postprocessing should only be applied on a ROI \
            around the user scribbles
        """
        model_checkpoint_path = ""
        self.model = None
        self.use_roi_post_processing = use_roi_post_processing
        super().__init__(
            path=model_checkpoint_path,
            network=self.model,
            type=monailabel.interfaces.tasks.infer.InferType.SCRIBBLES,
            labels="Tumor",
            dimension=3,
            description="Basic interactive segmentation inference task.",
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        transforms = init_transforms_load_and_normalize()

        return transforms

    def post_transforms(self, data=None) -> Sequence[Callable]:
        transforms = [
            monai.transforms.ToNumpyd(keys=["CT", "pred"], allow_missing_keys=True),
        ]

        # apply GraphCut on the model output with the CT image as the pairwise term
        if np.count_nonzero(data["label"]) > 0:
            if self.use_roi_post_processing:
                transforms += [
                    FallBackTransform(
                        [
                            RegionOfInterestTransform(
                                [
                                    MakeISegUnaryd(
                                        image="CT",
                                        logits="pred",
                                        scribbles="label",
                                        unary="unary",
                                    ),
                                    ApplyGraphCutOptimisationd(
                                        unary="unary",
                                        pairwise="CT",
                                        post_proc_label="pred",
                                    ),
                                ],
                                cropping_keys=["pred", "CT", "label"],
                                cropping_mask_key="label",
                                post_transform_map={"pred": 0},
                                seed_value=3,
                                roi_factor=3.5,
                                handle_missing_bbox="raise",
                            ),
                            RegionOfInterestTransform(
                                [
                                    MakeISegUnaryd(
                                        image="CT",
                                        logits="pred",
                                        scribbles="label",
                                        unary="unary",
                                    ),
                                    ApplyGraphCutOptimisationd(
                                        unary="unary",
                                        pairwise="CT",
                                        post_proc_label="pred",
                                    ),
                                ],
                                cropping_keys=["pred", "CT", "label"],
                                cropping_mask_key="proposal_discrete",
                                post_transform_map={"pred": 0},
                                seed_value=1,
                                roi_factor=3.5,
                            ),
                        ],
                        listening_errors=[ValueError],
                    ),
                ]
            else:
                transforms += [
                    ConvertBinaryToTwoChannels(keys=["pred"]),
                    ApplyGraphCutOptimisationd(
                        unary="pred", pairwise="CT", post_proc_label="pred"
                    ),
                ]
        else:
            transforms += [monai.transforms.CopyItemsd(keys="label", names="pred")]

        transforms += [
            monailabel.transform.post.Restored(keys="pred", ref_image="CT"),
            EnsurePrediction(pred_key="pred", input_key="label"),
            # manually insert the extension to inform the writer
            lambda d: dict({"result_extension": ".nii.gz"}, **d),
            SaveImageToFiled(
                "pred",
                "./inferlog/",
                file_prefix="pred_gc_",
                prefix_key="image",
                meta_dict_key="CT_meta_dict",
            ),
        ]
        return transforms

    def inferer(self, data=None) -> monai.inferers.Inferer:
        return DummyInferer()

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        if np.count_nonzero(data["label"]) > 0:
            data = MakeLikelihoodFromScribblesGMMd(
                image="CT", scribbles="label", post_proc_label="pred", num_mixtures=10
            )(data)
        return data

    def is_valid(self) -> bool:
        return True
