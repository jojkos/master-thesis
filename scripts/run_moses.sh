#!/usr/bin/env bash

MOSES_PATH="/home/jonas/mosesdecoder"
LANG_FROM="cs"
LANG_TO="de"
CORPUS_PATH="/mnt/g/Clouds/DPbigFiles/WMT17/newsCommentaryCsDe"
CORPUS_NAME="nc11"
TEST_CORPUS_PATH="/mnt/g/Clouds/DPbigFiles/WMT17/allTogetherPreprocessedMOSES"  # already tokenized and cleaned..
TEST_CORPUS_NAME="corpus.tc.dev2000"
TOOLS_PATH="/home/jonas/mosesdecoder/tools"
MAX_LENGTH=15
FULL_PATH=${CORPUS_PATH}"/"${CORPUS_NAME}
TOKENIZED_PATH=${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized"

cd ${MOSES_PATH}

if [ $1 = "--tokenize" ] || [ $1 = "--all" ] || [ $1 = "--preprocess" ]
then
    printf "tokenizing..\n\n"
    scripts/tokenizer/tokenizer.perl -no-escape -l ${LANG_FROM} \
        < ${FULL_PATH}"."${LANG_FROM} \
        > ${TOKENIZED_PATH}"."${LANG_FROM}
    scripts/tokenizer/tokenizer.perl -no-escape -l ${LANG_TO} \
        < ${FULL_PATH}"."${LANG_TO} \
        > ${TOKENIZED_PATH}"."${LANG_TO}
fi

if [ $1 = "--clean" ] || [ $1 = "--all" ] || [ $1 = "--preprocess" ]
then
    printf "cleaning..\n\n"
    scripts/training/clean-corpus-n.perl \
        ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized" ${LANG_FROM} ${LANG_TO} \
        ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned" 1 ${MAX_LENGTH}
fi

if [ $1 = "--truecase" ] || [ $1 = "--all" ] || [ $1 = "--preprocess" ]
then
    printf "truecaser training..\n\n"
    scripts/recaser/train-truecaser.perl \
        --model ${CORPUS_PATH}"/truecase-model."${LANG_FROM} \
        --corpus ${TOKENIZED_PATH}"."${LANG_FROM}
    scripts/recaser/train-truecaser.perl \
        --model ${CORPUS_PATH}"/truecase-model."${LANG_TO} \
        --corpus ${TOKENIZED_PATH}"."${LANG_TO}

    printf "truecasing..\n\n"
    scripts/recaser/truecase.perl \
        --model ${CORPUS_PATH}"/truecase-model."${LANG_FROM} \
        < ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned."${LANG_FROM} \
        > ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned.truecased."${LANG_FROM}
    scripts/recaser/truecase.perl \
        --model ${CORPUS_PATH}"/truecase-model."${LANG_TO} \
        < ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned."${LANG_TO} \
        > ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned.truecased."${LANG_TO}
fi

if [ $1 = "--train" ] || [ $1 = "--all" ]
then
    printf "language model training..\n\n"
    mkdir ${CORPUS_PATH}/languagemodel
    bin/lmplz -o 3 \
        < ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned.truecased."${LANG_TO} \
        > ${CORPUS_PATH}"/languagemodel/"${CORPUS_NAME}".arpa."${LANG_TO}

    printf "binarizing for faster loading..\n\n"
    bin/build_binary ${CORPUS_PATH}"/languagemodel/"${CORPUS_NAME}".arpa."${LANG_TO} \
        ${CORPUS_PATH}"/languagemodel/"${CORPUS_NAME}".binary."${LANG_TO}

    printf "training model..\n\n"
    scripts/training/train-model.perl \
         -root-dir ${CORPUS_PATH}"/"train \
         -corpus ${CORPUS_PATH}"/"${CORPUS_NAME}"-tokenized.cleaned.truecased" \
         -f ${LANG_FROM} -e ${LANG_TO} \
         -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
         -lm 0:3:${CORPUS_PATH}"/languagemodel/"${CORPUS_NAME}".binary."${LANG_TO}:8 \
         -external-bin-dir ${TOOLS_PATH} -cores 4 -parallel -mgiza -mgiza-cpus 4
fi


if [ $1 = "--test" ] || [ $1 = "--all" ]
then
    printf "testing..\n\n"

    bin/moses \
    -f ${CORPUS_PATH}"/train/model/moses.ini" \
    < ${TEST_CORPUS_PATH}/${TEST_CORPUS_NAME}"."${LANG_FROM} \
    > ${TEST_CORPUS_PATH}/${TEST_CORPUS_NAME}".translated."${LANG_TO} \
    2> ${TEST_CORPUS_PATH}/${TEST_CORPUS_NAME}".out"

    scripts/generic/multi-bleu.perl \
    -lc ${TEST_CORPUS_PATH}/${TEST_CORPUS_NAME}"."${LANG_TO} \
    < ${TEST_CORPUS_PATH}/${TEST_CORPUS_NAME}".translated."${LANG_TO}
fi

