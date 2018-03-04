#!/bin/bash

HOME_FOLDER="/pub/tmp/xholcn01"
MAX_EPOCHS=0
EPOCHS_RANGE=2

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
         -q long.q@@gpu \
         -l gpu=1,mem_free=15G,ram_free=15G,disk_free=5G,tmp_free=5G \
         -pe smp 4 \
         newsCommentaryBPE.sh

    retval=$?
    if [ ${retval} -ne 0 ]
    then
        echo "qsub ended with error ${retval}"
        break
    fi
done
