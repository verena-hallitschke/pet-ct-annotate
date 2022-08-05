"""(Pre-)train the model."""
import argparse
import json
import logging
import sys
import os
import torch
import data.multimodaldatastore
import data.loader
from model.model import PETCTAnnotationModel
from model.tasks import PETCTAnnotationTrainTask
from model.deepigeos_network import P_Rnet3D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", help="Which dataset to use. Can be either tcia or autopet."
    )
    parser.add_argument(
        "metadata_directory",
        help="Directory of the metadata.csv file of the TCIA dataset.",
    )
    parser.add_argument(
        "-a",
        "--annotations_dir",
        default=None,
        help="Root directory of the preprocessed annotation files. Only Required for the tcia dataset",
    )
    parser.add_argument(
        "-m", "--model_dir", nargs="?", default=".", help="Directory of the models."
    )
    parser.add_argument(
        "-d",
        "--device",
        nargs="?",
        default="cuda",
        help="Pytorch device to use for training.",
    )
    parser.add_argument(
        "-t",
        "--train_mode",
        nargs="?",
        default="rp",
        help="What network should be trained. 'rp': Both, ''p': P-Net,'r': R-Net.",
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        default="./train_config.json",
        help="Model and training params.",
    )
    return parser.parse_args()


def train():
    arguments = parse_args()
    with open(arguments.config) as f:
        config = json.load(f)
    print(
        f"""Starting training with {arguments.dataset} data and cuda {torch.cuda.is_available()}
        Deive count: {torch.cuda.device_count()}
        CUDA_VISIBLE_DEVICES {os.environ.get("CUDA_VISIBLE_DEVICES")}
        """
    )
    dataset_dataframe = data.loader.load_dataset(
        arguments.dataset,
        arguments.metadata_directory,
        split_type="train",
        annotations_dir=arguments.annotations_dir,
    )

    dataset = data.multimodaldatastore.MultimodalDatastore(
        dataset_dataframe,
        ["CT", "PET"],
        id_key="id",
        label_key="Segmentation",
        convert_to_nifti=False,
    )

    # ------ Training the preprocessing network ------
    if "p" in arguments.train_mode:
        print("Start Training of P-Net")
        p_net_path = os.path.join(arguments.model_dir, config["p_net_name"])
        if not os.path.exists(p_net_path):
            os.mkdir(p_net_path)
        train_task_pre = PETCTAnnotationTrainTask(
            P_Rnet3D(c_in=config["c_in_pnet"], c_blk=config["c_blk"]),
            p_net_path,
            config={
                "max_epochs": config["epochs"],
                "device": arguments.device,
                "train_batch_size": config["train_batch_size"],
                "val_batch_size": config["val_batch_size"],
            },
            amp=arguments.device == "cuda",
        )
        train_task_pre({"dataset": "Dataset"}, dataset)

        torch.cuda.empty_cache()  # explicitly free cache once between the two training steps

    # ------ Training the refinement network ------
    if "r" in arguments.train_mode:
        print("Start Training of R-Net")
        r_net_path = os.path.join(arguments.model_dir, config["r_net_name"])
        if not os.path.exists(r_net_path):
            os.mkdir(r_net_path)
        train_task_refine = PETCTAnnotationTrainTask(
            P_Rnet3D(c_in=config["c_in_rnet"], c_blk=config["c_blk"]),
            r_net_path,
            preprocess_network=P_Rnet3D(
                c_in=config["c_in_pnet"], c_blk=config["c_blk"]
            ),
            preprocess_network_path=os.path.join(
                arguments.model_dir, config["p_net_name"], "train_01", "model.pt"
            ),
            config={
                "max_epochs": config["epochs"],
                "device": arguments.device,
                "train_batch_size": config["train_batch_size"],
                "val_batch_size": config["val_batch_size"],
            },
            amp=arguments.device == "cuda",
        )

        train_task_refine({"dataset": "Dataset"}, dataset)


if __name__ == "__main__":
    train()
