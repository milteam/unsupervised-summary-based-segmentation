#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

if [ -z "$1" ]; then
    echo "Error: Please provide a path as the first argument to the data you want to process"
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
RUN_DIR="$SCRIPT_DIR/runs/naive_run_$RUN_START"

if [ ! -d "$SCRIPT_DIR/runs" ]; then
    mkdir "$SCRIPT_DIR/runs"
fi

rm -rf "$RUN_DIR"
mkdir $RUN_DIR

python3 "$PYTHON_SCRIPT" naive -i "$1" -dt "$2"
