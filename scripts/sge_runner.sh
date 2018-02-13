#!/bin/bash

HOME_FOLDER="/pub/tmp/xholcn01"
MAX_EPOCHS=100
EPOCHS_RANGE=50

for (( i=0; i <= MAX_EPOCHS; i+=EPOCHS_RANGE ))
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
         -o /pub/tmp/xholcn01/translation.out.${INITIAL_EPOCH} \
         -e /pub/tmp/xholcn01/translation.err.${INITIAL_EPOCH} \
         -q long.q@@gpu \
         -l gpu=1,mem_free=4G,ram_free=4G,disk_free=2G,tmp_free=2G
done
