#!/usr/bin/env sh
if [ ! -d "./log" ];then
   mkdir ./log
else
   :
fi
LOG=./log/log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=~/anaconda3/bin
nohup $PYDIR/python -B train.py 2>&1 | tee $LOG&


