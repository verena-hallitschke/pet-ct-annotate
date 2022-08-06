"""Model tasks for the MONAILabel server."""
from multiprocessing.sharedctypes import Value
import os
from typing import Callable, Dict, Optional, Sequence, Union, List

import numpy as np
import torch.optim
import torch.nn
import monai.inferers
import monai.transforms
import monailabel.tasks.train.basic_train
from monailabel.tasks.train.basic_train import Context
import monailabel.interfaces.tasks.infer
from monailabel.scribbles.transforms import (
    ApplyCRFOptimisationd,
    MakeISegUnaryd,
    ApplyGraphCutOptimisationd,
)

from transforms.utility import (
    AffineCopy,
    ConvertBinaryToTwoChannels,
    RandomRepeatTransform,
    FallBackTransform,
)
from transforms.distance_transform import GeodesicUserInputTransform
from transforms.mock_transform import CreateMockMask, CreateBackgroundMockMask
from transforms.network_transform import NetworkForwardTransform
from transforms.roi_transform import RegionOfInterestTransform
from transforms.logging_transform import SaveImageToFiled
from transforms.ensure_annotation_transform import EnsurePrediction

ALLOW_PROPOSAL = True

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


def init_transforms_preprocess_network(network, path, device="cuda"):
    """
    Initialize the transforms relating to the preprocessing network.

    :param network: Module of the network to be used.
    :param path: Path of the network parameters to be loaded.
    :param device: torch device to perform the calculation on.
    """
    return [
        NetworkForwardTransform(
            network,
            path,
            input_keys=[None, "CT", "PET", "annotation_fg", "annotation_bg",],
            device=device,
            output_key="proposal",
        ),
        monai.transforms.Activationsd(keys="proposal", sigmoid=True),
        monai.transforms.CopyItemsd(keys="proposal", names="proposal_discrete"),
        monai.transforms.AsDiscreted(keys=["proposal_discrete"], threshold=0.5),
        monai.transforms.ToNumpyd(keys="proposal_discrete"),
    ]


def init_transforms_geodesic_input(
    foreground_info: Dict, background_info: Dict, fallback_value=1
):
    """
    Initialize the geodesic transforms for the input.

    :param foreground_info: A dict containing info about the foreground input. Must contain the keys
        ["input_key", "input_value"], which are used for the geodesic transformation. Can contain
        the keys ["roi_key", "roi_value"], which are used to determine the region of interest for
        the calculations.
    :param background_info: A dict containing info about the background input with the same
        structure as the info for the foreground.
    """
    foreground_input_key = foreground_info["input_key"]
    foreground_value = foreground_info["input_value"]
    foreground_roi_key = (
        foreground_info["roi_key"]
        if "roi_key" in foreground_info
        else foreground_input_key
    )
    foreground_roi_value = (
        foreground_info["roi_value"]
        if "roi_value" in foreground_info
        else foreground_value
    )
    background_input_key = background_info["input_key"]
    background_value = background_info["input_value"]
    background_roi_key = (
        background_info["roi_key"]
        if "roi_key" in background_info
        else background_input_key
    )
    background_roi_value = (
        background_info["roi_value"]
        if "roi_value" in background_info
        else background_value
    )
    return [
        FallBackTransform(
            [
                RegionOfInterestTransform(
                    GeodesicUserInputTransform(
                        image_key="CT",
                        scribbles_mask_key=foreground_input_key,
                        output_key="annotation_fg",
                        seed_value=foreground_value,
                    ),
                    cropping_keys=["CT", foreground_input_key],
                    cropping_mask_key=foreground_roi_key,
                    post_transform_map={"annotation_fg": np.max},
                    seed_value=foreground_roi_value,
                    handle_missing_bbox="raise",
                ),
                RegionOfInterestTransform(
                    GeodesicUserInputTransform(
                        image_key="CT",
                        scribbles_mask_key=foreground_input_key,
                        output_key="annotation_fg",
                        seed_value=foreground_value,
                    ),
                    cropping_keys=["CT", foreground_input_key],
                    cropping_mask_key=foreground_roi_key,
                    post_transform_map={"annotation_fg": np.float32(10)},
                    seed_value=foreground_roi_value,
                    handle_missing_bbox="constant",
                ),
            ],
            listening_errors=[ValueError],
        ),
        AffineCopy("CT", "annotation_fg"),
        FallBackTransform(
            [
                RegionOfInterestTransform(
                    GeodesicUserInputTransform(
                        image_key="CT",
                        scribbles_mask_key=background_input_key,
                        output_key="annotation_bg",
                        seed_value=background_value,
                    ),
                    cropping_keys=["CT", background_input_key],
                    cropping_mask_key=background_roi_key,
                    post_transform_map={"annotation_bg": np.max},
                    seed_value=background_roi_value,
                    handle_missing_bbox="raise",
                ),
                RegionOfInterestTransform(
                    GeodesicUserInputTransform(
                        image_key="CT",
                        scribbles_mask_key=background_input_key,
                        output_key="annotation_bg",
                        seed_value=background_value,
                    ),
                    cropping_keys=["CT", background_input_key],
                    cropping_mask_key=background_roi_key,
                    post_transform_map={"annotation_bg": np.float32(10)},
                    seed_value=background_roi_value,
                    handle_missing_bbox="constant",
                ),
            ],
            listening_errors=[ValueError],
        ),
        AffineCopy("CT", "annotation_bg"),
    ]


class PETCTAnnotationInferTask(monailabel.interfaces.tasks.infer.InferTask):
    """MONAILabel entry point for inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        models_path: str,
        postprocessing: str = "direct",
        preprocess_network: torch.nn.Module = None,
        preprocess_network_path: str = None,
        use_roi_post_processing: bool = True,
    ) -> None:
        """Create the Infertask with the specified postprocessing step.

        :param model: the network to run inference on.
        :param models_path: base path of the models.
        :param postprocessing: type of postprocessing, must be one of ["direct", "graphcut"]
        :param preprocess_network: optional network to use as preprocessing step.
        :param preprocess_network_path: path of the saved parameters of the preprocessing network.
        :param use_roi_post_processing: Whether postprocessing should only be applied on a ROI \
            around the user scribbles
        """
        # pylint: disable=too-many-arguments
        model_checkpoint_path = os.path.join(models_path, "train_01", "model.pt")
        self.model = model
        self.postprocessing = postprocessing
        self.preprocess_network = preprocess_network
        self.preprocess_network_path = preprocess_network_path
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
        transforms += init_transforms_geodesic_input(
            {"input_key": "label", "input_value": 3},
            {"input_key": "label", "input_value": 2},
        )
        transforms += [
            monai.transforms.EnsureTyped(
                ["CT", "PET", "annotation_fg", "annotation_bg"]
            ),
        ]
        if self.preprocess_network is not None:
            transforms += init_transforms_preprocess_network(
                self.preprocess_network, self.preprocess_network_path
            )
        return transforms

    def post_transforms(self, data=None) -> Sequence[Callable]:
        transforms = [
            monai.transforms.ToNumpyd(keys=["CT", "pred"]),
        ]
        if self.postprocessing == "direct":
            # directly convert the network output to a segmentation
            transforms += [
                monai.transforms.AsDiscreted(keys=["pred"], threshold=0.5),
            ]
        elif self.postprocessing == "graphcut":
            # apply GraphCut on the model output with the CT image as the pairwise term
            if self.use_roi_post_processing:
                transforms += [
                    monai.transforms.CopyItemsd(
                        ["CT_meta_dict"], names=["pred_meta_dict"]
                    ),
                    monai.transforms.EnsureChannelFirstd(keys=["pred"]),
                    FallBackTransform(
                        [
                            RegionOfInterestTransform(
                                [
                                    ConvertBinaryToTwoChannels(keys=["pred"]),
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
                                    ConvertBinaryToTwoChannels(keys=["pred"]),
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

        elif self.postprocessing == "crf":

            transforms += [
                # Copy CT metadata to pred
                monai.transforms.CopyItemsd(["CT_meta_dict"], names=["pred_meta_dict"]),
                monai.transforms.EnsureChannelFirstd(keys=["pred"]),
                MakeISegUnaryd(
                    image="CT", logits="pred", scribbles="label", unary="unary",
                ),
            ]

            if self.use_roi_post_processing:
                transforms += [
                    RegionOfInterestTransform(
                        ApplyCRFOptimisationd(
                            unary="unary", pairwise="CT", post_proc_label="pred"
                        ),
                        cropping_keys=["unary", "CT"],
                        cropping_mask_key="label",
                        post_transform_map={"pred": 0},
                        seed_value=3,
                        roi_factor=3.5,
                    ),
                ]
            else:
                transforms += [
                    ApplyCRFOptimisationd(
                        unary="unary", pairwise="CT", post_proc_label="pred"
                    ),
                ]

            transforms += [
                monai.transforms.ToNumpyd(keys=["pred"], dtype=np.float32),
            ]

        transforms += [
            monailabel.transform.post.Restored(keys="pred", ref_image="CT"),
            EnsurePrediction(pred_key="pred", input_key="label"),
            # manually insert the extension to inform the writer
            lambda d: dict({"result_extension": ".nii.gz"}, **d),
            SaveImageToFiled(
                "pred",
                "./inferlog/",
                file_prefix="pred_",
                prefix_key="image",
                meta_dict_key="CT_meta_dict",
            ),
        ]
        return transforms

    def inferer(self, data=None) -> monai.inferers.Inferer:
        return monai.inferers.SimpleInferer()

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        if (
            self.preprocess_network is not None
            and torch.count_nonzero(data["annotation_fg"] != np.float32(10)) == 0
            and torch.count_nonzero(data["annotation_bg"] != np.float32(10)) == 0
        ):
            if ALLOW_PROPOSAL:
                # no user input, return proposal as output
                data["pred"] = data["proposal"]
            else:
                data["pred"] = np.zeros_like(data["CT"])
            return data

        inferer = self.inferer()
        network = self._get_network(device)
        image_ct = data["CT"].to(device)
        image_pet = data["PET"].to(device)
        annotation_fg = data["annotation_fg"].to(device)
        annotation_bg = data["annotation_bg"].to(device)
        if self.preprocess_network is not None:
            proposal = data["proposal"]
        if convert_to_batch:
            image_ct = image_ct[None]
            image_pet = image_pet[None]
            annotation_fg = annotation_fg[None]
            annotation_bg = annotation_bg[None]
            if self.preprocess_network is not None:
                proposal = proposal[None]
        inferer_args = [image_ct, image_pet, annotation_fg, annotation_bg]
        if self.preprocess_network is not None:
            inferer_args += [proposal]
        data["pred"] = inferer(None, network, *inferer_args)[:, 0]
        data["pred"] = monai.transforms.Activations(sigmoid=True)(data["pred"])
        if convert_to_batch:
            data["pred"] = data["pred"][0]
        return data

    def is_valid(self) -> bool:
        return True


class PETCTAnnotationTrainTask(monailabel.tasks.train.basic_train.BasicTrainTask):
    """MONAILabel entry point for training."""

    def __init__(
        self,
        model: torch.nn.Module,
        models_path: str,
        preprocess_network: torch.nn.Module = None,
        preprocess_network_path: str = None,
        **kwargs,
    ) -> None:
        """Create the TrainingTask.

        :param model: the network to train.
        :param models_path: base path of the models.
        :param preprocess_network: optional network to use as preprocessing step.
        :param preprocess_network_path: path of the saved parameters of the preprocessing network.
        """
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), 1e-3, (0.9, 0.999))
        self.preprocess_network = preprocess_network
        self.preprocess_network_path = preprocess_network_path
        super().__init__(
            model_dir=models_path,
            description="Basic segmentation training task.",
            **kwargs,
        )

    def network(self, context: Context):
        return self.model

    def optimizer(self, context: Context):
        return self.optim

    def loss_function(self, context: Context):
        return monai.losses.DiceCELoss(sigmoid=True)

    def train_pre_transforms(self, context: Context) -> Sequence[Callable]:
        transforms = init_transforms_load_and_normalize()
        transforms += [
            monai.transforms.RandGaussianNoised(keys=["CT", "PET"], prob=0.3),
        ]
        transforms += [
            lambda d: dict({"mask": torch.zeros_like(d["CT"])}, **d),
            lambda d: dict({"mask_bg": torch.zeros_like(d["CT"])}, **d),
            RandomRepeatTransform(
                CreateMockMask(
                    label_mask_key="label",
                    image_key="CT",
                    output_key="mask",
                    seed_value=1,
                    result_fill_value=1,
                ),
                output_key="mask",
                repeat_times_min=1,
                repeat_times_max=3,
            ),
            RandomRepeatTransform(
                CreateBackgroundMockMask(
                    label_mask_key="label",
                    image_key="CT",
                    output_key="mask_bg",
                    seed_value=1,
                    result_fill_value=1,
                ),
                output_key="mask_bg",
                repeat_times_min=0,
                repeat_times_max=1,
            ),
        ]
        transforms += init_transforms_geodesic_input(
            {"input_key": "mask", "input_value": 1},
            {"input_key": "mask_bg", "input_value": 1},
        )
        transforms += [
            monai.transforms.EnsureTyped(
                ["CT", "PET", "annotation_fg", "annotation_bg"]
            ),
        ]
        if self.preprocess_network is not None:
            transforms += init_transforms_preprocess_network(
                self.preprocess_network,
                self.preprocess_network_path,
                self._config["device"],
            )
        return transforms

    def train_post_transforms(self, context: Context) -> Sequence[Callable]:
        return [
            monai.transforms.Activationsd(keys="pred", sigmoid=True),
            monai.transforms.AsDiscreted(keys=["pred", "label"], to_onehot=2),
        ]

    def val_pre_transforms(self, context: Context) -> Sequence[Callable]:
        return self.train_pre_transforms(context)

    def train_inferer(self, context: Context):
        return monai.inferers.SimpleInferer()

    def val_inferer(self, context: Context):
        return monai.inferers.SimpleInferer()

    def train_key_metric(self, context: Context):
        return {
            "train_dice": monai.handlers.MeanDice(
                output_transform=monai.handlers.from_engine(["pred", "label"]),
                include_background=False,
            )
        }

    def val_key_metric(self, context: Context):
        return {
            "val_dice": monai.handlers.MeanDice(
                output_transform=monai.handlers.from_engine(["pred", "label"]),
                include_background=False,
            )
        }

    def _get_batch_keys(self) -> List[str]:
        batch_keys = ["CT", "PET", "annotation_fg", "annotation_bg"]
        if self.preprocess_network is not None:
            batch_keys += ["proposal"]
        return batch_keys

    def _create_trainer(self, context: Context):
        train_handlers = self.train_handlers(context)
        if context.local_rank == 0:
            train_handlers.append(
                monai.handlers.checkpoint_saver.CheckpointSaver(
                    save_dir=context.output_dir,
                    save_dict={self._model_dict_key: context.network},
                    save_interval=self._train_save_interval,
                    save_final=True,
                    final_filename=self._final_filename,
                    save_key_metric=True,
                    key_metric_filename=f"train_{self._key_metric_filename}"
                    if context.evaluator
                    else self._key_metric_filename,
                )
            )

        self._load_checkpoint(context, train_handlers)

        batch_keys = self._get_batch_keys()

        return monai.engines.trainer.SupervisedTrainer(
            device=context.device,
            max_epochs=context.max_epochs,
            train_data_loader=self.train_data_loader(context),
            network=context.network,
            optimizer=context.optimizer,
            loss_function=self.loss_function(context),
            prepare_batch=PrepareBatchExtraInputNoImage(batch_keys),
            inferer=self.train_inferer(context),
            amp=self._amp,
            postprocessing=self._validate_transforms(
                self.train_post_transforms(context), "Training", "post"
            ),
            key_train_metric=self.train_key_metric(context),
            train_handlers=train_handlers,
            iteration_update=self.train_iteration_update(context),
            event_names=self.event_names(context),
        )

    def _create_evaluator(self, context: Context):
        evaluator = None
        if context.val_datalist and len(context.val_datalist) > 0:
            val_hanlders = self.val_handlers(context)
            if context.local_rank == 0:
                val_hanlders.append(
                    monai.handlers.checkpoint_saver.CheckpointSaver(
                        save_dir=context.output_dir,
                        save_dict={self._model_dict_key: context.network},
                        save_key_metric=True,
                        key_metric_filename=self._key_metric_filename,
                        n_saved=self._n_saved,
                    )
                )

            batch_keys = self._get_batch_keys()

            evaluator = monai.engines.evaluator.SupervisedEvaluator(
                device=context.device,
                val_data_loader=self.val_data_loader(context),
                network=context.network,
                inferer=self.val_inferer(context),
                prepare_batch=PrepareBatchExtraInputNoImage(batch_keys),
                postprocessing=self._validate_transforms(
                    self.val_post_transforms(context), "Validation", "post"
                ),
                key_val_metric=self.val_key_metric(context),
                additional_metrics=self.val_additional_metrics(context),
                val_handlers=val_hanlders,
                iteration_update=self.val_iteration_update(context),
                event_names=self.event_names(context),
            )
        return evaluator


class PrepareBatchExtraInputNoImage(monai.engines.utils.PrepareBatch):
    # pylint: disable=too-few-public-methods
    """
    Customized prepare_batch for trainer or evaluator that support extra input data for network,
    while also omitting the "image" data.
    Extra items are specified by the `extra_keys` parameter.

    Args:
        extra_keys: if a string or list provided, every item is the key
            of extra data in current batch,
            and will pass the extra data to the `network(*args)` in order.
            If a dictionary is provided, every `{k, v}` pair is the key
            of extra data in current batch,
            `k` is the param name in network, `v` is the key of extra data in current batch,
            and will pass the `{k1: batch[v1], k2: batch[v2], ...}` as kwargs to the network.

    """

    def __init__(self, extra_keys: Union[Sequence[str], Dict[str, str]]) -> None:
        self.extra_keys = extra_keys

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        **kwargs,
    ):
        """
        Args `batchdata`, `device`, `non_blocking` refer to the ignite API:
        https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html.
        `kwargs` supports other args for `Tensor.to()` API.

        """
        label = batchdata["label"].to(
            device=device, non_blocking=non_blocking, **kwargs
        )
        args_ = []
        kwargs_ = {}

        def _get_data(key: str):
            data = batchdata[key]
            return (
                data.to(device=device, non_blocking=non_blocking, **kwargs)
                if isinstance(data, torch.Tensor)
                else data
            )

        if isinstance(self.extra_keys, (str, list, tuple)):
            for k in self.extra_keys:
                args_.append(_get_data(k))
        elif isinstance(self.extra_keys, dict):
            for k, v in self.extra_keys.items():
                kwargs_.update({k: _get_data(v)})

        return None, label, tuple(args_), kwargs_
