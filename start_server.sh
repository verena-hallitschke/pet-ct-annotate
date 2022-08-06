#!/bin/bash

# Arguments
# $1: GPU that should be used (default: 5)
# $2: Port number (default: 8000)
# $3: Path to sample app (default ./src)

while getopts ":hp:d:m:l:" opt; do
    case $opt in
        p) PORT="$OPTARG";;
        d) CUDA_DEVICE="$OPTARG";;
        m) MONAI_LOC="$OPTARG";;
        l) PROJECT_PATH="$OPTARG";;
        h) echo Shell script to start a multimodal MONAILabel server. 
        echo start_server.sh [-h] [-p PORT] [-d CUDA_DEVICE] [-m MONAI_PATH] [-l PROJECT_PATH] DATA_LOCATION LABEL_LOCATION
        echo -e "\n"
        echo ARGUMENTS:
        echo
        echo -e "\t DATA_LOCATION \t\t Path to the autopet dataset."
        echo -e "\t LABEL_LOCATION \t Path to the location where the labels will be saved."
        echo
        echo OPTIONS:
        echo
        echo -e "\t -h \t\t\t Display this help menu."
        echo -e "\t -p PORT \t\t Port on which the server starts, defaults to 8000."
        echo -e "\t -d CUDA_DEVICE \t CUDA device number on which the inference is performed, defaults to 5. The devices can be checked with the nvidia-smi command."
        echo -e "\t -m MONAI_PATH \t\t Path to the MONAILabel executable, defaults to \"./MONAILabelMultimodality/monailabel/scripts/monailabel\"."
        echo -e "\t -l PROJECT_PATH \t Path to the pet-ct-annotate source folder, defaults to \"./src\"."
        exit 0
        ;;
        \?) echo "Unknown command line option -$OPTARG" >&2
        exit 1;;
    esac
    
    case $OPTARG in
        -*) echo "Invalid argument for option $opt."
        exit 1
        ;;
    esac
done

if [ -z "$CUDA_DEVICE" ]
then
        CUDA_DEVICE=5
fi

if [ -z "$PORT" ]
then
        PORT=8000
fi

if [ -z "$PROJECT_PATH" ]
then
        PROJECT_PATH="./src"
fi

if [ -z "$MONAI_LOC" ]
then
        MONAI_LOC="./MONAILabelMultimodality/monailabel/scripts/monailabel"
fi

if [ -z "${@:$OPTIND+1:1}" ]
then
    echo "Missing positional argument \"label location\""
    exit 1
else
        LABEL_LOC="${@:$OPTIND+1:1}"
fi

if [ -z "${@:$OPTIND:1}" ]
then
    echo "Missing positional argument \"dataset location\""
    exit 1
else
    DATA_LOC="${@:$OPTIND:1}"
fi

echo $CUDA_DEVICE $MONAI_LOC $PORT $PROJECT_PATH $DATA_LOC $LABEL_LOC

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $MONAI_LOC start_server -p $PORT -a $PROJECT_PATH -s $DATA_LOC --conf label_path $LABEL_LOC --conf dataset autopet