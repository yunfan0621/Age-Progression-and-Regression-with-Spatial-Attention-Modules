#!/bin/bash

set -x
set -e

time CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ./datasets \
                                           --save_suffix test \
                                           --epochs 50 \
                                           --gpu_ids 0
