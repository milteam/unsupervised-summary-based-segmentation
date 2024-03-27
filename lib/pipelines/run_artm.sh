#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

if [ -z "$1" ]; then
    echo "Error: Please provide a path as the first argument."
    exit 1
fi

if [ ! -e "$1" ]; then
    echo "Error: The argument '$1' is not a valid path to dataset"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Please provide dataset type"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/main.py"
RUN_START=$(date +'%Y%m%d_%H%M%S')
RUN_DIR="$SCRIPT_DIR/runs/artm_run_$RUN_START"
DATASET_TRAIN="$1/train"
DATASET_TEST="$1/test"

if [ ! -d "$DATASET_TRAIN" ]; then
    echo "Error: '$DATASET_TRAIN' is not a valid directory."
    exit 1
fi

if [ ! -d "$DATASET_TEST" ]; then
    echo "Error: '$DATASET_TEST' is not a valid directory."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/runs" ]; then
    mkdir "$SCRIPT_DIR/runs"
fi

rm -rf "$RUN_DIR"
mkdir $RUN_DIR

TRAIN_VECTORIZED="$RUN_DIR/train_vectorized"
TEST_VECTORIZED="$RUN_DIR/test_vectorized"
MODEL_PATH="$RUN_DIR/model"
PREDICTS_PATH="$RUN_DIR/predicts"

python3 "$PYTHON_SCRIPT" artm vectorize -i "$DATASET_TRAIN" -o "$TRAIN_VECTORIZED" -c train -dt "$2"
python3 "$PYTHON_SCRIPT" artm vectorize -i "$DATASET_TEST" -o "$TEST_VECTORIZED" -c test -d "$TRAIN_VECTORIZED/vocab.train.txt" -dt "$2"
python3 "$PYTHON_SCRIPT" artm train -o "$MODEL_PATH" -i "$TRAIN_VECTORIZED" -dt "$2"
python3 "$PYTHON_SCRIPT" artm predict -m "$MODEL_PATH" -i "$TEST_VECTORIZED" -o "$PREDICTS_PATH" -d "$TRAIN_VECTORIZED/batches/dictionary.dict" -dt "$2"
python3 "$PYTHON_SCRIPT" artm scores -i "$DATASET_TEST" -p "$PREDICTS_PATH/predicts.pickle" -dt "$2"