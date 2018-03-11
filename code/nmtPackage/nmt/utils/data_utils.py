# coding: utf-8

import logging
import random
import os
import errno
import shutil

import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import text_to_word_sequence
from bs4 import BeautifulSoup
from gensim.models.wrappers import FastText

logger = logging.getLogger(__name__)


def prepare_folder(path, clear):
    """

    Args:
        path (str): path to the folder
        clear (bool): whether to clear the folder if it already exists

    """
    logger.debug("preparing folder {}".format(path))

    if clear:
        try:
            logger.info("deleting folder {}".format(path))
            shutil.rmtree(path)
        except:
            pass

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger.error("problem with creating folder {}: {}".format(path, e))
            raise
        else:
            logger.debug("folder {} already exists".format(path))


def prepare_folders(paths, clear):
    """

    Args:
        paths (str[]): list of paths to the folders
        clear (bool): whether to clear the folders if they already exists

    """
    for path in paths:
        prepare_folder(path, clear)


def read_file_to_lines(path, max_lines):
    """

    Loads lines from a given file up to max_lines

    Args:
        path: path to the file
        max_lines: maximum number of lines to be read from file, None means all of them

    Returns: list of string lines read from the file

    """
    logger.info("reading {} into lines...".format(path))
    with open(path, encoding="utf-8") as file_handler:
        lines = file_handler.read().split("\n")[:max_lines]

    return lines


def load_embedding_weights(path, words, limit=None):
    """

    Loads pretrained embedding weights from given path
    
    Args:
        path: Path to fast text embeddings
        words: dictionary of words (e.g. {0: "PAD", 1: "cat", 2: "dog"}),
            word on index 0 is considered to be PADDING and its weights are set to 0
        limit: maximum number of loaded embeddings (lines)
    
    Returns:
        list of weights in the same order as are words

    """
    logger.debug("load_embedding_weights from {} for {} words"
                 .format(path, len(words)))

    # don't know how to limit size of vocabulary with FastText
    # loading the whole model takes too much time
    # TODO use FastText (.bin) for final version, because it can generate vectors for OOV words from ngrams
    model = FastText.load_fasttext_format(path)

    # model = KeyedVectors.load_word2vec_format(path, limit=limit)

    # model.get_keras_embedding()

    # Get dimension
    dim = model.vector_size

    logger.info("getting embedding weights for each word")
    weights = []
    oov_words = 0
    for index, word in words.items():
        if index == 0:
            # Set zero weight for padding symbol
            weight = np.zeros(dim)
        elif word in model:
            # https://radimrehurek.com/gensim/models/wrappers/fasttext.html
            # The word can be out-of-vocabulary as long as ngrams for the word are present.
            # For words with all ngrams absent, a KeyError is raised.
            weight = model[word]
        else:
            # logging.warning("out of vocabulary word: {}".format(word))
            oov_words += 1
            # Init random weights for out of vocabulary word
            # TODO are the values in range (-1, 1)?
            # weight = np.random.uniform(low=-1.0, high=1.0, size=dim)

            # TODO in final version change to fastText model
            weight = model.seeded_vector(random.random())
            # https://www.quora.com/How-does-fastText-output-a-vector-for-a-word-that-is-not-in-the-pre-trained-model

        weights.append(weight)

    logging.warning("{} oov words".format(oov_words))
    logging.info("weights loaded")
    return np.asarray(weights)


def get_bucket_ix(seq_length, bucket_range):
    """

    Returns index of a bucket for a sequence with q given length when bucket range is bucket_range

    Args:
        seq_length: lengh of sequence
        bucket_range: range of bucket

    Returns: index of a bucket

    """
    return seq_length // bucket_range + (1 if seq_length % bucket_range != 0 else 0)


def split_to_buckets(x_sequences, y_sequences, bucket_range=3, x_max_len=None, y_max_len=None, bucket_min_size=10):
    """

    Split list of sequences to list of buckets where each bucket is a list of word sequences
    with similar length (based on the bucket_range size e.g.
    with bucket_size=3 sequences with length 1-3 falls in same bucket)
    
    Args:
        x_sequences: one list of sequences
        y_sequences: another list of sequences
        bucket_range: size of one bucket (how big range of sequence lengths should fall into one bucket)
        x_max_len: optional max length of X sequences
        y_max_len: optional max length of y sequences
        bucket_min_size: minimal size of a bucket. If its lower, than the bucket gets merged with other bucket
    
    Returns:
        dict of buckets each with the Y and y list of similary long sequences and their max length
        
    """
    logger.info("splitting sequences to buckets with range {}".format(bucket_range))

    assert len(x_sequences) == len(y_sequences)

    if not x_max_len:
        x_max_len = max(len(seq) for seq in x_sequences)
    if not y_max_len:
        y_max_len = max(len(seq) for seq in y_sequences)

    all_max_len = max(x_max_len, y_max_len)

    logger.debug("x_max_len = {}, y_max_len = {}".format(x_max_len, y_max_len))

    num_buckets = get_bucket_ix(all_max_len, bucket_range)

    buckets = {}

    logger.debug("max num buckets = {}".format(num_buckets))

    for i in range(len(x_sequences)):
        x_seq = x_sequences[i]
        y_seq = y_sequences[i]
        x_len = len(x_seq)
        y_len = len(y_seq)

        max_len = max(x_len, y_len)
        bucket = get_bucket_ix(max_len, bucket_range)

        if bucket not in buckets:
            buckets[bucket] = {"x_word_seq": [], "y_word_seq": [], "x_max_seq_len": 0, "y_max_seq_len": 0}

        buckets[bucket]["x_word_seq"].append(x_seq)
        buckets[bucket]["y_word_seq"].append(y_seq)
        buckets[bucket]["x_max_seq_len"] = max(buckets[bucket]["x_max_seq_len"], x_len)
        buckets[bucket]["y_max_seq_len"] = max(buckets[bucket]["y_max_seq_len"], y_len)

    # merge buckets lower then bucket_min_size for optimization
    # so we don't run fit method over really small input lists
    logger.debug("bucket_min_size={}".format(bucket_min_size))
    delete_ixs = []
    bucket_ixs = sorted(buckets.keys())
    merge_bucket_ix = 0
    for ix, bucket_ix in enumerate(bucket_ixs):
        bucket = buckets[bucket_ix]
        if len(bucket["x_word_seq"]) < bucket_min_size:
            if ix < len(bucket_ixs) - 1:
                merge_bucket_ix = bucket_ixs[ix + 1]
            elif ix > 1:  # if last bucket, then it has to be merged to the lower one
                merge_bucket_ix = bucket_ixs[ix - 1]

            if merge_bucket_ix > 0 and merge_bucket_ix not in delete_ixs:
                logger.info("bucket {} is too small, merging with bucket {}".format(bucket_ix, merge_bucket_ix))
                delete_ixs.append(bucket_ix)

                buckets[merge_bucket_ix]["x_max_seq_len"] = max(buckets[merge_bucket_ix]["x_max_seq_len"],
                                                                bucket["x_max_seq_len"])
                buckets[merge_bucket_ix]["y_max_seq_len"] = max(buckets[merge_bucket_ix]["y_max_seq_len"],
                                                                bucket["y_max_seq_len"])
                buckets[merge_bucket_ix]["x_word_seq"] += bucket["x_word_seq"]
                buckets[merge_bucket_ix]["y_word_seq"] += bucket["y_word_seq"]

    for ix in delete_ixs:
        del buckets[ix]

    logger.debug("created {} buckets".format(len(buckets)))
    for bucket in buckets:
        logger.debug("bucket {} with xmaxlen {}, ymaxlen {} has {} sequences".format(
            bucket, buckets[bucket]["x_max_seq_len"], buckets[bucket]["y_max_seq_len"],
            len(buckets[bucket]["x_word_seq"]))
        )

    return buckets


def tokenize(lines):
    """

    Tokenizes lines using keras's text_to_word_sequence

    Args:
        lines (str[]): array of lines

    Returns: list of splitted word sequences

    """
    logger.info("tokenizing lines...")
    # TODO use tokenization from Moses so its same as for Moses baseline model
    word_seq = [text_to_word_sequence(x) for x in lines]

    return word_seq


def split_lines(lines):
    """

    Splits each line on ' ' character

    Args:
        lines (str[]): array of lines

    Returns: returns array of splitted sequences

    """
    word_seq = []

    for line in lines:
        splitted = line.split(" ")
        # skip empty lines
        if len(splitted) > 0:
            word_seq.append(splitted)

    return word_seq


def convert_xmlset_to_text(path):
    """

    Converts data sets in xml (sgm) (e.g. http://www.statmt.org/wmt17/translation-task.html Development and Test sets
    to text only format (one line, one sequence).
    Creates new file with same name and .lines extension

    Args:
        path: path to the file

    """
    with open(path, encoding="utf-8") as f:
        data = f.read()

    soup = BeautifulSoup(data, "xml")

    with open(path + ".lines", "w", encoding="utf-8") as f_out:
        for doc in soup.find_all("doc"):
            for seg in doc.find_all("seg"):
                f_out.write(seg.text + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     weights = load_embedding_weights("G:/Clouds/DPbigFiles/facebookVectors/facebookPretrained-wiki.cs.vec",
    #                            {0:"PAD", 1:"kostka", 2:"pes", 3:"UNK"}, limit=1000)

    # x_word_seq = [
    #     ["1"],
    #     ["1", "2"],
    #     ["1", "2", "3"],
    #     ["1", "2", "3", "4"],
    #     ["1", "2", "3", "4", "5"],
    #     ["1", "2", "3", "4", "5"],
    #     ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # ]
    #
    # y_word_seq = [
    #     ["1"],
    #     ["1", "2"],
    #     ["1", "2"],
    #     ["1", "2", "3", "4"],
    #     ["1", "2", "3", "4", "5"],
    #     ["1", "2", "3", "4", "5", "6", "7"],
    #     ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # ]
    # buckets = split_to_buckets(x_word_seq, y_word_seq, 2, bucket_min_size=2)
    #
    # for bucket in buckets:
    #     print(bucket, buckets[bucket])
