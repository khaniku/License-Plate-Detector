#!/usr/bin/env bash

# Necessary exports
export PYTHONPATH=${PWD}/src:$PYTHONPATH

# Training
python ./train.py

# Predict
python ./predict.py

