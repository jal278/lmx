#!/bin/bash

# Run the one_max experiment on GPU 0
CUDA_VISIBLE_DEVICES=0 python order_parents.py one_max &

# Run the leading_ones experiment on GPU 1
CUDA_VISIBLE_DEVICES=1 python order_parents.py leading_ones &
