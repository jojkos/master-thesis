#!/bin/bash
#$ -N translation
#$ -o /tmp/xholcn01/translation.out
#$ -e /tmp/xholcn01/translation.err
#$ -q long.q@@gpu
#$ -l gpu=1,ram_free=2GB,tmp_free=2GB

TMP_FOLDER="G:/Dropbox/FIT/DP/code/nmt/scripts/tmp/xholcn01"
#HOME_FOLDER="/pub/tmp/xholcn01"
HOME_FOLDER="G:/Dropbox/FIT/DP/code"
MODEL_FILE="model_weights.h5"

if [ ! -d ${TMP_FOLDER} ]
then
    mkdir ${TMP_FOLDER}
fi

cd ${TMP_FOLDER}

python "${HOME_FOLDER}/main.py" \
    --training_dataset "${HOME_FOLDER}/data/europarl-v7.cs-en" \
    --test_dataset "${HOME_FOLDER}/data/anki_ces-eng" \
    --model_folder ${TMP_FOLDER} --model_file ${MODEL_FILE} \
    --log_folder "${TMP_FOLDER}/logs" \
    --source_lang "cs" --target_lang "en" \
    --embedding_path "G:/Clouds/DPbigFiles/facebookVectors/facebookPretrained-wiki.cs.vec" --embedding_dim 300 \
    --max_embedding_num 20000 \
    --latent_dim 300 --validation_split 0.1 \
    --max_source_vocab_size 20000 --max_target_vocab_size 20000\
    --epochs 1\
    --batch_size 1\
    --num_samples -1

cp "${TMP_FOLDER}/translation.out" "${HOME_FOLDER}/translation.out"
cp "${TMP_FOLDER}/translation.err" "${HOME_FOLDER}/translation.out"
mv "${TMP_FOLDER}/${MODEL_FILE}" "${HOME_FOLDER}/${MODEL_FILE}"