#!/usr/bin/env bash

function run_py(){

    code_path=./
    for file in $(ls)
    do
      if [[ $file =~ .py ]]
        then
          python $code_path$file
          if [ $? -eq 0 ]
            then
              echo run $code_path$file succeed in $python_version
            else
              echo run $code_path$file failed in $python_version
              exit -1
          fi
      fi
    done


}

## python3
python_version=python3
source activate base
cd ..
python setup.py install
cd ./examples
run_py

#python2
python_version=python2
source activate py27
cd ..
python setup.py install
cd ./examples
run_py
echo "all examples run succeed in python2.7"


echo "all examples run succeed in python3.6"

echo "all examples run succeed in python2.7 and python3.6"