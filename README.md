# pet-ct-annotate

## Set-up

### Python environment
Create miniconda/Anaconda environment using the following command:
```bash
conda env create -f environment.yml
```

For venv the `requirements.txt` file can be used:
```bash
pip install -r requirements.txt
```

Note: The package `GeodisTK` requires Microsoft C++ Build Tools on Windows devices. It can be downloaded [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Note: Usage of Anaconda or Miniconda is recommended since the package `cudatoolkit` is necessary for GPU execution.

### Git

This project uses a submodule. Either clone using
```bash
git clone --recursive <link>
```

or initialize the submodules after cloning:
```bash
git submodule update --init --recursive
```

On Windows machines, set git to automatically convert line endings:

```bash
git config --global core.autocrlf true
```

### MONAILabel Server


Use the file [`start_server.sh`](./start_server.sh):

```bash
conda activate petct
cd ./pet-ct-annotate/
pip install ./MONAILabelMultimodality/
start_server.sh <path/to/autopet> <path/to/label/folder>
```

The script supports the following command line arguments:

```bash
Shell script to start a multimodal MONAILabel server.
start_server.sh [-h] [-p PORT] [-d CUDA_DEVICE] [-m MONAI_PATH] [-l PROJECT_PATH] DATA_LOCATION LABEL_LOCATION


ARGUMENTS:

         DATA_LOCATION           Path to the autopet dataset.
         LABEL_LOCATION          Path to the location where the labels will be saved.

OPTIONS:

         -h                      Display this help menu.
         -p PORT                 Port on which the server starts, defaults to 8000.
         -d CUDA_DEVICE          CUDA device number on which the inference is performed, defaults to 5. The devices can be checked with the nvidia-smi command.
         -m MONAI_PATH           Path to the MONAILabel executable, defaults to "./MONAILabelMultimodality/monailabel/scripts/monailabel".
         -l PROJECT_PATH         Path to the pet-ct-annotate source folder, defaults to "./src".
```

#### Model

The model used in this project is a multimodal early fusion version of DeepIGeoS (see [Resources](#resources)). It consists of a proposal (P-Net) and a refinement network (R-Net). The model weights are located in [src/models](./src/models/). The model was trained using the autopet dataset (see [Resources](#resources)) using only the samples labeled as non-small-cell lung carcinoma.

### 3DSlicer

The fork of [MONAILabel](https://github.com/verena-hallitschke/MONAILabelMultimodality), is based on commit [ad2e081a](https://github.com/Project-MONAI/MONAILabel/commit/ad2e081ae69e88cb4f3c112a071c9b5a435b04f5) and contains three different versions of the 3D Slicer plugin:
* [MONAILabel](./MONAILabelMultimodality/plugins/slicer/MONAILabel/): The original plugin
* [MONAILabelMultimodality](./MONAILabelMultimodality/plugins/slicer/MONAILabelMultimodality/): A plugin that supports the MultimodalDatastore and is capable of showing both modalities at the same time
* [MONAILabelSingleView](./MONAILabelMultimodality/plugins/slicer/MONAILabelSingleView/): A plugin that supports the MultimodalDatastore and is capable of showing one modality at a time. Modalities can be swapped by pressing the button in the UI.

Follow the instructions in [MONAILabel Slicer Plugin](MONAILabelMultimodality/plugins/slicer/README.md) to install the Slicer Plugin in Developer Mode.

Connect to the MONAILabel Server in the Slicer MONAILabel Module.

## Linting

The code can be formatted using the following commands:

```bash
black src
black tests
```

## Developers

* [tcsch](https://github.com/tcsch)
* [realptkkit](https://github.com/realptkkit)
* [verena-hallitschke](https://github.com/verena-hallitschke)

## Resources


* [3D Slicer](https://www.slicer.org/): Fedorov A., Beichel R., Kalpathy-Cramer J., Finet J., Fillion-Robin J-C., Pujol S., Bauer C., Jennings D., Fennessy F.M., Sonka M., Buatti J., Aylward S.R., Miller J.V., Pieper S., Kikinis R. [3D Slicer as an Image Computing Platform for the Quantitative Imaging Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3466397/). Magn Reson Imaging. 2012 Nov;30(9):1323-41. PMID: 22770690. PMCID: PMC3466397.
* [MONAILabel](https://monai.io/): Andres Diaz-Pinto, Sachidanand Alle, Alvin Ihsani, Muhammad Asad, Vishwesh Nath, Fernando P??rez-Garc??a, Pritesh Mehta, Wenqi Li, Holger R. Roth, Tom Vercauteren, Daguang Xu, Prerna Dogra, Sebastien Ourselin, Andrew Feng, and M. Jorge Cardoso. [Monai label: A framework for ai-assisted interactive labeling of 3d medical images](https://arxiv.org/pdf/2203.12362.pdf), 2022.
* [MONAI](https://monai.io/): MONAI Consortium. (2022). [MONAI: Medical Open Network for AI](https://github.com/Project-MONAI/MONAI) (Version 0.9.1) [Computer software].
* DeepIGeoS and [GeodisTK](https://github.com/taigw/GeodisTK): Wang, Guotai, et al. DeepIGeoS: [A deep interactive geodesic framework for medical image segmentation](https://ieeexplore.ieee.org/document/8370732). TPAMI, 2018.
* [This implementation of DeepIGeoS](https://github.com/HITLAB-DeepIGeoS/DeepIGeoS)
* [autopet dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287): Gatidis S, Kuestner T. [A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) [Dataset]](https://doi.org/10.7937/gkr0-xv29). The Cancer Imaging Archive, 2022. 
