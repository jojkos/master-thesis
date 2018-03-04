#!/bin/bash
#$ -V
#$ -o /pub/tmp/xholcn01/translation.out
#$ -e /pub/tmp/xholcn01/translation.err

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

# delete old model in current /tmp if there is from some old run
if [ -f "${TMP_FOLDER}/${MODEL_FILE}" ]
then
    echo "deleting old ${TMP_FOLDER}/${MODEL_FILE} model"
    rm  "${TMP_FOLDER}/${MODEL_FILE}"
fi

# if model already exists from some older run, it needs to be copied to tmp folder to be used
# because every job can run on different pc with different tmp folder?
if [ -f "${HOME_FOLDER}/${MODEL_FILE}" ]
then
    echo "copying model from ${HOME_FOLDER}/${MODEL_FILE}"
    cp "${HOME_FOLDER}/${MODEL_FILE}" "${TMP_FOLDER}/${MODEL_FILE}"
fi

python3 "${HOME_FOLDER}/main.py" \
    --${MODE} \
    --training_dataset "${HOME_FOLDER}/data/news-commentary-v12.cs-en-tokenized.truecased.cleaned" \
    --test_dataset "${HOME_FOLDER}/data/newstest2015-csen" \
    --model_folder ${TMP_FOLDER} --model_file ${MODEL_FILE} \
    --log_folder "${HOME_FOLDER}/logs" \
    --source_lang "cs" --target_lang "en" \
    --max_source_vocab_size 35000 --max_target_vocab_size 35000 \
    --initial_epoch "${INITIAL_EPOCH}" --epochs "${EPOCHS}" \
    --batch_size 64 --num_units 512 \
    --bucketing True \
    --find_gpu True


mv "${TMP_FOLDER}/${MODEL_FILE}" "${HOME_FOLDER}/${MODEL_FILE}"

