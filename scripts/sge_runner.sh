#!/bin/bash

HOME_FOLDER="/pub/tmp/xholcn01"
MAX_EPOCHS=20
EPOCHS_RANGE=2

export PYTHONPATH=${PYTHONPATH}:/homes/eva/xh/xholcn01/.local/lib/python3.6/site-packages/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/share/cuda-9.0.176/lib64:/usr/local/share/cuda-9.0.176/extras/CUPTI/lib64
export CUDA_HOME=/usr/local/share/cuda-9.0.176

for (( i=2; i <= MAX_EPOCHS; i+=EPOCHS_RANGE ))
do
    export INITIAL_EPOCH=${i}
    export EPOCHS=$((INITIAL_EPOCH + EPOCHS_RANGE))

    if [ ${i} = ${MAX_EPOCHS} ]
    then
        export MODE=evaluate
    else
        export MODE=train
    fi

    qsub -sync y \
         -N xholcnNMT${INITIAL_EPOCH} \
         -q long.q@dellgpu2,long.q@facegpu2,long.q@supergpu1,long.q@supergpu2,long.q@supergpu5,long.q@supergpu6,long.q@supergpu7 \
         -l gpu=1,mem_free=18G,ram_free=18G,disk_free=5G,tmp_free=5G \
         experiments/finalTest2/finalTest2.sh
    
    retval=$?
    if [ ${retval} -ne 0 ]
    then
        echo "qsub ended with error ${retval}"
        break
    fi
done
