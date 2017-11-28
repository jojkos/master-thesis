#!/bin/bash
#$ -N translation
#$ -o /homes/kazi/xholcn01/translation.out
#$ -e /homes/kazi/xholcn01/translation.err
#$ -q long.q@@gpu
#$ -l gpu=1,ram_free=2GB,tmp_free=2GB

python3 "/homes/kazi/xholcn01/translation.py"