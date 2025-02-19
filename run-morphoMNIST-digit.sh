#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name morphoMNIST-digit-test"
PYARGS="$PYARGS --data_name morphoMNIST"
PYARGS="$PYARGS --label_names label"
PYARGS="$PYARGS --data_path $DATA/data/morphoMNIST"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --hparams hparams/hparams_morphoMNIST.json"
PYARGS="$PYARGS --device cuda:0"

PYARGS="$PYARGS --omega0 10.0"
PYARGS="$PYARGS --omega1 0.50"
PYARGS="$PYARGS --omega2 0.01"
PYARGS="$PYARGS --dist_outcomes bernoulli"
PYARGS="$PYARGS --dist_mode discriminate"
#PYARGS="$PYARGS --checkpoint_classifier /path/to/trained/classifier"

PYARGS="$PYARGS --max_epochs 200"
PYARGS="$PYARGS --batch_size 32"
PYARGS="$PYARGS --checkpoint_freq 2"

python main.py $PYARGS
