#!/bin/bash
cd maddpg/experiments/

python train_v3.py --num-agents 4 --num-landmarks 3 --rsrn-type WSM --network fully-connected --num-episodes 150000 --exp-name test