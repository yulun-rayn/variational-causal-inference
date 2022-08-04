#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name train-epoch-1000"
PYARGS="$PYARGS --data_path $DATA/datasets/marson_prepped.h5ad" #sciplex_prepped.h5ad
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --gpu 0" #PYARGS="$PYARGS --cpu"

PYARGS="$PYARGS --outcome_dist normal"
PYARGS="$PYARGS --dist_mode match"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --batch_size 64"

python main_train.py $PYARGS
