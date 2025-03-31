#!/bin/bash

clean=false
if [[ $1 == "clean" ]]; then
  clean=true
fi

HOME_DIR=$(dirname ${BASH_SOURCE[0]})

set -e

source $(dirname $CONDA_EXE)/deactivate

set +e
conda env remove -n qrc_paper -y
set -e

conda env create -n qrc_paper -f environment.yml -y
source $(dirname $CONDA_EXE)/activate qrc_paper

pip install .[dev,test]

cd $HOME_DIR/experiments/function_approx
qrc_run_expt
qrc_postprocess
if [[ $clean ]]; then
  ls -1 *].npz | while read line; do rm "$line"; touch "$line"; done
fi
qrc_eval_tasks
qrc_generate_interp_data

cd ../image_classification
qrc_run_expt
qrc_postprocess
if [[ $clean ]]; then
  ls -1 *].npz | while read line; do rm "$line"; touch "$line"; done
fi
qrc_eval_tasks
qrc_eval_classification
qrc_parse_log

cd $HOME_DIR

find ./figures/generate_*.py | xargs -n 1 python
