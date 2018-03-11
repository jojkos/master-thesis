#!/bin/bash
#$ -V
#$ -o /pub/tmp/xholcn01/experiments/openSubtitles1mil/translation.out
#$ -e /pub/tmp/xholcn01/experiments/openSubtitles1mil/translation.err

TMP_FOLDER="/tmp/xholcn01"
MAIN_FOLDER="/pub/tmp/xholcn01"
DATA_FOLDER="/pub/tmp/xholcn01/data"
EXPERIMENT_FOLDER="/pub/tmp/xholcn01/experiments/openSubtitles1mil"
MODEL_FILE="openSubtitles.h5"

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
if [ -f "${EXPERIMENT_FOLDER}/${MODEL_FILE}" ]
then
    echo "copying model from ${EXPERIMENT_FOLDER}/${MODEL_FILE}"
    cp "${EXPERIMENT_FOLDER}/${MODEL_FILE}" "${TMP_FOLDER}/${MODEL_FILE}"
fi

python3 "${MAIN_FOLDER}/main.py" \
    --${MODE} \
    --training_dataset "${DATA_FOLDER}/OpenSubtitles2018.cs-en.1mil-tokenized.truecased.cleaned.BPE" \
    --test_dataset "${DATA_FOLDER}/test2000FromOpenSubtitles2018" \
    --model_folder ${TMP_FOLDER} --model_file ${MODEL_FILE} \
    --log_folder "${EXPERIMENT_FOLDER}/logs" \
    --source_lang "cs" --target_lang "en" \
    --max_source_vocab_size 15000 --max_target_vocab_size 15000 \
    --initial_epoch "${INITIAL_EPOCH}" --epochs "${EPOCHS}" \
    --source_embedding_path "${DATA_FOLDER}/facebookPretrained-wiki.cs.bin" \
    --target_embedding_path "${DATA_FOLDER}/facebookPretrained-wiki.en.bin" \
    --batch_size 256 --num_units 512 \
    --bucketing True \
    --find_gpu True && mv "${TMP_FOLDER}/${MODEL_FILE}" "${EXPERIMENT_FOLDER}/${MODEL_FILE}"

