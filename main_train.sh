#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name train-epoch-1000"
PYARGS="$PYARGS --artifact_path $DATA/artifact"

PYARGS="$PYARGS --data $DATA/datasets/marson_prepped_ood.h5ad"
PYARGS="$PYARGS --covariate_keys celltype donor stim"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --loss_dist normal"

python vci/main_train.py $PYARGS
