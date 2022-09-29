#!bin/bash
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg m2tr.yaml
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg vit.yaml
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg multiatt.yaml
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg xception.yaml
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg resnet50.yaml
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg efficientnet.yaml
CUDA_VISIBLE_DEVICES=4,5,6 python run.py --cfg f3net.yaml
