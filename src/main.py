"""Entry point for MONAILabel."""
import logging
import os
import copy
from typing import Dict
import monailabel
import monailabel.interfaces.app
from monailabel.config import settings
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.interfaces.tasks.strategy import Strategy
import monailabel.interfaces.datastore
import model.tasks
import model.graph_cut_tasks
import model.model
from model.deepigeos_network import P_Rnet3D
import data.loader
import data.multimodaldatastore

logger = logging.getLogger(__name__)


class PETCTApp(monailabel.interfaces.app.MONAILabelApp):
    """App class loaded by MONAILabel."""

    def __init__(self, app_dir: str, studies: str, conf: Dict[str, str]) -> None:
        self.models_dir = os.path.join(app_dir, "models")
        self.label_path = conf.get("label_path", ".")
        self.use_roi_post_processing = conf.get("roi_post_processing", True)
        self._modality_keys = ["CT", "PET"]
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="PET-CT-Annotate",
            description="Interactive annotation model for PET/CT images.",
        )
        os.makedirs(self.models_dir, exist_ok=True)

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "PET-CT-Annotation-Direct": model.tasks.PETCTAnnotationInferTask(
                P_Rnet3D(c_in=5, c_blk=4),
                models_path=os.path.join(self.models_dir, "RNet"),
                postprocessing="direct",
                preprocess_network=P_Rnet3D(c_in=2, c_blk=4),
                preprocess_network_path=os.path.join(
                    self.models_dir, "PNet", "train_01", "model.pt"
                ),
            ),
            "PET-CT-Annotation-GC": model.tasks.PETCTAnnotationInferTask(
                P_Rnet3D(c_in=5, c_blk=4),
                models_path=os.path.join(self.models_dir, "RNet"),
                postprocessing="graphcut",
                preprocess_network=P_Rnet3D(c_in=2, c_blk=4),
                preprocess_network_path=os.path.join(
                    self.models_dir, "PNet", "train_01", "model.pt"
                ),
            ),
            "PET-CT-Annotation-CRF": model.tasks.PETCTAnnotationInferTask(
                P_Rnet3D(c_in=5, c_blk=4),
                models_path=os.path.join(self.models_dir, "RNet"),
                postprocessing="crf",
                preprocess_network=P_Rnet3D(c_in=2, c_blk=4),
                preprocess_network_path=os.path.join(
                    self.models_dir, "PNet", "train_01", "model.pt"
                ),
            ),
            "GraphCutOnly": model.graph_cut_tasks.GraphCutTask(),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "PET-CT-Annotation": model.tasks.PETCTAnnotationTrainTask(
                model.model.PETCTAnnotationModel(), self.models_dir
            )
        }

    def init_datastore(self) -> data.multimodaldatastore.MultimodalDatastore:
        dataframe = data.loader.load_dataset(
            self.conf.get("dataset", "tcia"), self.studies, **self.conf
        )
        logger.info(dataframe)
        return data.multimodaldatastore.MultimodalDatastore(
            dataframe, self._modality_keys, "id", label_path=self.label_path
        )

    def init_strategies(self) -> Dict[str, Strategy]:
        return {"next": NextSampleStrategy()}

    def next_sample(self, request):
        strategy = request.get("strategy")
        strategy = strategy if strategy else "next"

        task = self._strategies.get(strategy)
        return {"id": task(request, self.datastore())}

    def infer(self, request, datastore=None):
        """
        Run Inference for an exiting pre-trained model.

        Args:
            request: JSON object which contains `model`, `image`, `params` and `device`
            datastore: Datastore object.
                If None then use default app level datastore to save labels if applicable

            For example::

                {
                    "device": "cuda"
                    "model": "segmentation_spleen",
                    "image": "file://xyz",
                    "save_label": "true/false",
                    "label_tag": "original"
                }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        request_model = request.get("model")  #
        if not request_model:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                "Model is not provided for Inference Task",
            )

        task = self._infers.get(request_model)
        if not task:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                f"Inference Task is not Initialized. There is no model '{request_model}' available",
            )

        request = copy.deepcopy(request)
        request["description"] = task.description

        image_id = request["image"]
        if isinstance(image_id, str):
            datastore = datastore if datastore else self.datastore()
            for modality_key in self._modality_keys:
                request[modality_key] = datastore.get_image_modality_uri(
                    image_id, modality_key
                )

                if os.path.isdir(request[modality_key]):
                    logger.info("Input is a Directory; Consider it as DICOM")
                    logger.info(os.listdir(request[modality_key]))
                    request[modality_key] = [
                        os.path.join(f, request[modality_key])
                        for f in os.listdir(request[modality_key])
                    ]

            logger.debug("Image => %s", request["image"])
        else:
            request["save_label"] = False

        if self._infers_threadpool:

            def run_infer_in_thread(t, r):
                return t(r)

            f = self._infers_threadpool.submit(run_infer_in_thread, t=task, r=request)
            result_file_name, result_json = f.result(
                request.get("timeout", settings.MONAI_LABEL_INFER_TIMEOUT)
            )
        else:
            result_file_name, result_json = task(request)

        label_id = None
        if result_file_name and os.path.exists(result_file_name):
            tag = request.get(
                "label_tag", monailabel.interfaces.datastore.DefaultLabelTag.ORIGINAL
            )
            save_label = request.get("save_label", False)
            if save_label:
                label_id = datastore.save_label(image_id, result_file_name, tag, {})
            else:
                label_id = result_file_name

        return {
            "label": label_id,
            "tag": monailabel.interfaces.datastore.DefaultLabelTag.ORIGINAL,
            "file": result_file_name,
            "params": result_json,
        }


class NextSampleStrategy(Strategy):
    """Sampling strategy to deterministically return the next unlabeled sample."""

    def __init__(self):
        super().__init__("Next sample strategy")

    def __call__(self, request, datastore: monailabel.interfaces.datastore.Datastore):
        unlabeled_images = datastore.get_unlabeled_images()
        if len(unlabeled_images) > 0:
            return unlabeled_images[0]
        return None
