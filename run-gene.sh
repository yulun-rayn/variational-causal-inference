#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name marson-test"
PYARGS="$PYARGS --data_name gene"
PYARGS="$PYARGS --data_path $DATA/datasets/marson_prepped.h5ad" #sciplex_prepped.h5ad
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --hparams hparams/hparams_gene.json"
PYARGS="$PYARGS --device cuda:0"

PYARGS="$PYARGS --omega0 1.0"
PYARGS="$PYARGS --omega1 1.7"
PYARGS="$PYARGS --omega2 0.1"
PYARGS="$PYARGS --dist_outcomes normal"
PYARGS="$PYARGS --dist_mode match"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --batch_size 64"
PYARGS="$PYARGS --eval_mode classic"

python main.py $PYARGS
