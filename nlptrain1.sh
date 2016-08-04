#!/bin/sh
#
#This is an example script example.sh
#
#These commands set up the Grid Environment for your job:
# Request gpu queue, default queue is cpu and then nothing has to be specified
#PBS -N 'PTB_lstm_seqlen100hiddensize2x1024trainsize2to17' 
#PBS -q gpu

#load torch
module load torch/7
module load cuda/7.5
module load gcc/6.1.0

#change to home dir
cd /home/thj92/DeepLearningNLP/

#start training
th recurrent-language-model.lua
