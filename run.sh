#!/bin/bash


if [ -z ${CUDA_VISIBLE_DEVICES} ]; then
    export CUDA_VISIBLE_DEVICES="0"
fi

python main.py
