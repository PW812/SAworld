#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29502}
PYTHON=${PYTHON:-/xs-train-nas/workspace_cpw/conda/wam_cpw/bin/python}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"$PYTHON" -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
