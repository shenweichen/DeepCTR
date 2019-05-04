#!/usr/bin/env bash

cd ..
pip install deepctr -e .
cd ./examples
code_path=./

python $code_path"run_classification_criteo.py"
python $code_path"run_classification_criteo_hash.py"
python $code_path"run_regression_movielens.py"
python $code_path"run_multivalue_movielens.py"
python $code_path"run_multivalue_movielens_hash.py"
python $code_path"run_dien.py"
python $code_path"run_din.py"


echo "examples run done!!"