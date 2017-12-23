#!/bin/bash
#$ -N neural machine translation
#$ -o /pub/tmp/xholcn01/translation.out
#$ -e /pub/tmp/xholcn01/translation.err
#$ -q long.q@@gpu
#$ -l gpu=1,ram_free=2GB,tmp_free=2GB

#TMP_FOLDER="G:/Dropbox/FIT/DP/code/nmt/scripts/tmp/xholcn01"
TMP_FOLDER="/tmp/xhholcn01"
HOME_FOLDER="/pub/tmp/xholcn01"
#HOME_FOLDER="G:/Dropbox/FIT/DP/code"
LOG_FOLDER="logs"
MODEL_FILE="model_weights.h5"

if [ ! -d ${TMP_FOLDER} ]
then
    mkdir ${TMP_FOLDER}
fi

cd ${TMP_FOLDER} || {
  echo Not able to cd to ${TMP_FOLDER}
  exit 1
}

python "${HOME_FOLDER}/main.py" \
    --training_dataset "${HOME_FOLDER}/data/anki_ces-eng" \
    --test_dataset "${HOME_FOLDER}/data/OpenSubtitles2016-moses-10000.cs-en-tokenized.truecased.cleaned" \
    --model_folder ${TMP_FOLDER} --model_file ${MODEL_FILE} \
    --log_folder "${TMP_FOLDER}/${LOG_FOLDER}" \
    --source_lang "cs" --target_lang "en" \
    --embedding_path "/pub/tmp/xholcn01/facebookPretrained-wiki.cs.vec" \
    --embedding_dim 300 \
    --latent_dim 150 --validation_split 0.1 \
    --max_source_vocab_size 20000 --max_target_vocab_size 20000 \
    --epochs 50 \
    --batch_size 64 \

mv "${TMP_FOLDER}/translation.out" "${HOME_FOLDER}/translation.out"
mv "${TMP_FOLDER}/translation.err" "${HOME_FOLDER}/translation.out"
mv "${TMP_FOLDER}/${LOG_FOLDER}" "${HOME_FOLDER}/${LOG_FOLDER}"
mv "${TMP_FOLDER}/${MODEL_FILE}" "${HOME_FOLDER}/${MODEL_FILE}"
