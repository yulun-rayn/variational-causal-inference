#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name celebA-classifier"
PYARGS="$PYARGS --data_name celebA"
PYARGS="$PYARGS --data_path $DATA/data/celebA"
PYARGS="$PYARGS --artifact_path $DATA/artifact/classifier"
PYARGS="$PYARGS --hparams hparams/hparams_celebA.json"
PYARGS="$PYARGS --device cuda:0"

PYARGS="$PYARGS --max_epochs 200"
PYARGS="$PYARGS --batch_size 64"

python main_classifier.py $PYARGS
