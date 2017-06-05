#!/bin/sh
echo "TRAINING GEDI model on GPU $1"
CUDA_VISIBLE_DEVICES=$1 python training_and_eval/train_vgg16.py
echo "TESTING GEDI model on GPU $1"
CUDA_VISIBLE_DEVICES=$1 python training_and_eval/test_vgg16.py
