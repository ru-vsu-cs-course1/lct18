#!/bin/bash

CD=$(dirname "$(readlink -f "$0")")  # "

# export PYTHON_HOME=/opt/python3
export PYTHON_HOME="$CD/.venv"


export PATH=$PYTHON_HOME/bin:$PATH

export LD_LIBRARY_PATH=/opt/gdal-3.7.2/lib:$LD_LIBRARY_PATH
