#!/bin/bash
#$ -V

export PYTHONPATH=${PYTHONPATH}:/homes/eva/xh/xholcn01/.local/lib/python3.6/site-packages/

#TMP_FOLDER="G:/Clouds/DPbigFiles/WMT17/devSet/test"
TMP_FOLDER="/tmp/xholcn01"
HOME_FOLDER="/pub/tmp/xholcn01"
#HOME_FOLDER="G:/Dropbox/FIT/DP/code"
MODEL_FILE="model_weights.h5"

if [ ! -d ${TMP_FOLDER} ]
then
    mkdir ${TMP_FOLDER}
fi

cd ${TMP_FOLDER} || {
  echo Not able to cd to ${TMP_FOLDER}
  exit 1
}

python3 "${HOME_FOLDER}/main.py" \
    --${MODE} \
    --training_dataset "${HOME_FOLDER}/data/newstest2015-csen" \
    --test_dataset "${HOME_FOLDER}/data/newstest2015-csen" \
    --model_folder ${TMP_FOLDER} --model_file ${MODEL_FILE} \
    --log_folder "${HOME_FOLDER}/logs" \
    --source_lang "cs" --target_lang "en" \
    --max_source_vocab_size 15000 --max_target_vocab_size 15000 \
    --initial_epoch "${INITIAL_EPOCH}" --epochs "${EPOCHS}" \
    --batch_size 64 \

if [ ${MODE} = evaluate ]
then
  mv "${TMP_FOLDER}/${MODEL_FILE}" "${HOME_FOLDER}/${MODEL_FILE}"
fi
