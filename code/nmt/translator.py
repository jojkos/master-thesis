# coding: utf-8
# model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

import logging
import os
import argparse
import pickle
from time import time

import numpy as np
from nmt.utils import read_file_to_lines, load_embedding_weights, split_to_buckets, prepare_folders
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
import random

# TODO how to properly log
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Translator(object):
    # Special vocabulary symbols
    _PAD = "_PAD"
    _GO = "_GO"
    _EOS = "_EOS"
    _UNK = "_UNK"

    # pad is zero, because default value in the matrices is zero (np.zeroes)
    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, batch_size, bucketing, bucket_range, embedding_dim, embedding_path, epochs, eval_translation,
                 in_lang, latent_dim,
                 log_folder, max_in_vocab_size, max_out_vocab_size, model_file, model_folder, num_samples,
                 reverse_input,
                 target_lang, test_dataset, training_dataset, validaton_split):
        self.batch_size = batch_size
        self.bucketing = bucketing
        self.bucket_range = bucket_range
        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.epochs = epochs
        self.eval_translation = eval_translation
        self.in_lang = in_lang
        self.latent_dim = latent_dim
        self.log_folder = log_folder
        self.max_in_vocab_size = max_in_vocab_size
        self.max_out_vocab_size = max_out_vocab_size
        # self.model_file = model_file
        self.model_folder = model_folder
        self.model_weights_path = "{}{}".format(model_folder, model_file)
        self.num_samples = num_samples
        self.reverse_input = reverse_input
        self.target_lang = target_lang
        self.test_dataset_path = test_dataset
        self.training_dataset_path = training_dataset
        self.validation_split = validaton_split

        prepare_folders([self.log_folder, self.model_folder])

        self.training_dataset = self.prepare_training_dataset()
        self.test_dataset = self.prepare_testing_dataset()

        self.embedding_weights = None
        if not os.path.isfile(self.model_weights_path) and self.embedding_path:
            # load pretrained embeddings
            self.embedding_weights = load_embedding_weights(self.embedding_path,
                                                            self.training_dataset["x_ix_to_word"],
                                                            limit=self.max_in_vocab_size)

        self.model, self.encoder_model, self.decoder_model = self.define_models()

        self.model.summary()

        # logging for tensorboard
        self.tensorboard_callback = TensorBoard(log_dir="{}{}".format(self.log_folder, time()),
                                                write_graph=False)

        logger.info("compiling model...")
        # Run training
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           metrics=['acc'])

        if os.path.isfile(self.model_weights_path):
            logger.info("Loading model weights from file..")
            self.model.load_weights(self.model_weights_path)

    @staticmethod
    def tokenize(x_lines, y_lines):
        logger.info("tokenizing lines...")
        # TODO use tokenization from Moses so its same as for Moses baseline model
        x_word_seq = [text_to_word_sequence(x) for x in x_lines]
        y_word_seq = [[Translator._GO] + text_to_word_sequence(y) + [Translator._EOS] for y in y_lines]

        # Retrieving max sequence length for both source and target
        x_max_seq_len = max(len(seq) for seq in x_word_seq)
        y_max_seq_len = max(len(seq) for seq in y_word_seq)

        logger.info("Max sequence length for inputs: {}".format(x_max_seq_len))
        logger.info("Max sequence length for targets: {}".format(y_max_seq_len))

        return x_word_seq, y_word_seq, x_max_seq_len, y_max_seq_len

    @staticmethod
    def insert_symbol_to_vocab(vocab, symbol, index):
        """
        symbol can potentially (for instance as a result of tokenizing where _go and _eos are added to sequence)
        be already part of vocabulary, but we want it to be on specific index
        """

        if symbol in vocab:
            vocab.remove(symbol)

        vocab.insert(index, symbol)

        return vocab

    def get_vocabularies(self, x_word_seq, y_word_seq):
        logger.info("creating vocabularies...")
        # Creating the vocabulary set with the most common words
        # TODO how many most common words to use?
        dist = FreqDist(np.hstack(x_word_seq))
        x_vocab = dist.most_common(self.max_in_vocab_size)
        dist = FreqDist(np.hstack(y_word_seq))
        y_vocab = dist.most_common(self.max_out_vocab_size)

        # Creating an array of words from the vocabulary set,
        # we will use this array as index-to-word dictionary
        x_ix_to_word = [word[0] for word in x_vocab]
        # ADD special vocabulary symbols at the start
        Translator.insert_symbol_to_vocab(x_ix_to_word, Translator._PAD, Translator.PAD_ID)
        Translator.insert_symbol_to_vocab(x_ix_to_word, Translator._UNK, Translator.UNK_ID)
        x_ix_to_word = {index: word for index, word in enumerate(x_ix_to_word)}
        x_word_to_ix = {x_ix_to_word[ix]: ix for ix in x_ix_to_word}
        # TODO how to use pretrained embedding with these custom symbols
        # https://github.com/fchollet/keras/issues/6480
        # https://github.com/fchollet/keras/issues/3325
        x_vocab_len = len(x_ix_to_word)

        y_ix_to_word = [word[0] for word in y_vocab]
        Translator.insert_symbol_to_vocab(y_ix_to_word, Translator._PAD, Translator.PAD_ID)
        Translator.insert_symbol_to_vocab(y_ix_to_word, Translator._GO, Translator.GO_ID)
        Translator.insert_symbol_to_vocab(y_ix_to_word, Translator._EOS, Translator.EOS_ID)
        Translator.insert_symbol_to_vocab(y_ix_to_word, Translator._UNK, Translator.UNK_ID)
        y_ix_to_word = {index: word for index, word in enumerate(y_ix_to_word)}
        y_word_to_ix = {y_ix_to_word[ix]: ix for ix in y_ix_to_word}
        y_vocab_len = len(y_ix_to_word)

        result = (x_ix_to_word, x_word_to_ix, x_vocab_len,
                  y_ix_to_word, y_word_to_ix, y_vocab_len)

        logger.info("Number of samples: {}".format(len(x_word_seq)))
        logger.info("Number of input dictionary: {}".format(x_vocab_len))
        logger.info("Number of target dictionary: {}".format(y_vocab_len))

        return result

    def encode_sequences(self, x_word_seq, y_word_seq,
                         x_max_seq_len, y_max_seq_len,
                         x_word_to_ix, y_word_to_ix):
        """
        Take word sequences and convert them so that the model can be fit with them.
        Input words are just converted to integer index
        Target words are encoded to one hot vectors of target vocabulary length
        """
        logger.info("Encoding sequences...")

        y_vocab_len = len(y_word_to_ix)

        encoder_input_data = np.zeros(
            (len(x_word_seq), x_max_seq_len), dtype='float32')
        decoder_input_data = np.zeros(
            (len(x_word_seq), y_max_seq_len, y_vocab_len),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(x_word_seq), y_max_seq_len, y_vocab_len),
            dtype='float32')

        # prepare source sentences for embedding layer (encode to indexes)
        for i, seq in enumerate(x_word_seq):
            if self.reverse_input:  # for better results according to paper Sequence to seq...
                seq = seq[::-1]
            for t, word in enumerate(seq):
                if word in x_word_to_ix:
                    encoder_input_data[i][t] = x_word_to_ix[word]
                else:
                    encoder_input_data[i][t] = Translator.UNK_ID

        # encode target sentences to one hot encoding
        for i, seq in enumerate(y_word_seq):
            for t, word in enumerate(seq):
                if word in y_word_to_ix:
                    index = y_word_to_ix[word]
                else:
                    index = Translator.UNK_ID
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t][index] = 1

                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, index] = 1

        return encoder_input_data, decoder_input_data, decoder_target_data

    def prepare_training_dataset(self):
        x_file_path = "{}.{}".format(self.training_dataset_path, self.in_lang)
        x_lines = read_file_to_lines(x_file_path, self.num_samples)

        y_file_path = "{}.{}".format(self.training_dataset_path, self.target_lang)
        y_lines = read_file_to_lines(y_file_path, self.num_samples)

        x_word_seq, y_word_seq, x_max_seq_len, y_max_seq_len = Translator.tokenize(x_lines, y_lines)

        x_ix_to_word, x_word_to_ix, x_vocab_len, y_ix_to_word, y_word_to_ix, y_vocab_len = self.get_vocabularies(
            x_word_seq,
            y_word_seq
        )

        if self.bucketing:
            encoder_input_data = []
            decoder_input_data = []
            decoder_target_data = []

            buckets = split_to_buckets(x_word_seq, y_word_seq, self.bucket_range, x_max_seq_len, y_max_seq_len)

            for ix, bucket in buckets.items():
                enc_in, dec_in, dec_tar = self.encode_sequences(
                    bucket["x_word_seq"], bucket["y_word_seq"],
                    bucket["x_max_seq_len"], bucket["y_max_seq_len"],
                    x_word_to_ix, y_word_to_ix
                )

                encoder_input_data.append(enc_in)
                decoder_input_data.append(dec_in)
                decoder_target_data.append(dec_tar)
        else:
            encoder_input_data, decoder_input_data, decoder_target_data = self.encode_sequences(
                x_word_seq, y_word_seq,
                x_max_seq_len, y_max_seq_len,
                x_word_to_ix, y_word_to_ix
            )

            encoder_input_data = [encoder_input_data]
            decoder_input_data = [decoder_input_data]
            decoder_target_data = [decoder_target_data]

        return {
            "x_word_seq": x_word_seq, "y_word_seq": y_word_seq,
            "x_ix_to_word": x_ix_to_word, "x_word_to_ix": x_word_to_ix,
            "x_vocab_len": x_vocab_len, "y_ix_to_word": y_ix_to_word,
            "y_word_to_ix": y_word_to_ix, "y_vocab_len": y_vocab_len,
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def prepare_testing_dataset(self):
        """
        # vocabularies of test dataset has to be the same as of training set
        # otherwise embeddings would not correspond are use OOV
        # and y one hot encodings wouldnt correspond either

        Returns:

        """
        x_file_path = "{}.{}".format(self.training_dataset_path, self.in_lang)
        x_lines = read_file_to_lines(x_file_path, self.num_samples)

        y_file_path = "{}.{}".format(self.training_dataset_path, self.target_lang)
        y_lines = read_file_to_lines(y_file_path, self.num_samples)

        x_word_seq, y_word_seq, x_max_seq_len, y_max_seq_len = Translator.tokenize(x_lines, y_lines)

        encoder_input_data, decoder_input_data, decoder_target_data = self.encode_sequences(
            x_word_seq, y_word_seq,
            x_max_seq_len, y_max_seq_len,
            self.training_dataset["x_word_to_ix"], self.training_dataset["y_word_to_ix"]
        )

        return {
            "y_ix_to_word": self.training_dataset["y_ix_to_word"],
            "y_vocab_len": self.training_dataset["y_vocab_len"],
            "y_max_seq_len": y_max_seq_len,
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def define_models(self):
        logger.info("Creating models...")
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))

        if self.embedding_weights is not None:
            self.embedding_weights = [self.embedding_weights]  # Embedding layer wantes list as parameter
        # TODO trainable False or True?
        # TODO according to https://keras.io/layers/embeddings/
        # input dim should be +1 when used with mask_zero..is it correctly set here?
        embedding = Embedding(self.training_dataset["x_vocab_len"], self.embedding_dim,
                              weights=self.embedding_weights, mask_zero=True)
        embedding_outputs = embedding(encoder_inputs)

        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(embedding_outputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.training_dataset["y_vocab_len"]))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.training_dataset["y_vocab_len"], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

    def decode_sequence(self, input_seq, encoder_model, decoder_model,
                        y_ix_to_word, y_vocab_len, y_max_seq_len):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, y_vocab_len))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, Translator.GO_ID] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1). # TODO ? can the batch size be bigger?
        decoded_sentence = ""
        while True:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = y_ix_to_word[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_word == Translator._EOS:
                break

            decoded_sentence += sampled_word + " "

            if len(decoded_sentence) > y_max_seq_len:
                break

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, y_vocab_len))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    @staticmethod
    def encode_text_to_input_seq(text, word_to_ix):
        sequences = text_to_word_sequence(text)
        x = np.zeros((1, len(text)), dtype='float32')

        for i, seq in enumerate(sequences):
            if seq in word_to_ix:
                ix = word_to_ix[seq]
            else:
                ix = Translator.UNK_ID
            x[0][i] = ix

        return x

    def fit(self):
        logger.info("fitting the model...")
        for i in range(self.epochs):
            logger.info("Epoch {}".format(i + 1))

            for j in range(len(self.training_dataset["encoder_input_data"])):
                if self.bucketing:
                    logger.info("Bucket {}".format(j))

                self.model.fit(
                    [
                        self.training_dataset["encoder_input_data"][j],
                        self.training_dataset["decoder_input_data"][j]
                    ],
                    self.training_dataset["decoder_target_data"][j],
                    batch_size=self.batch_size,
                    epochs=1,
                    validation_split=self.validation_split,
                    callbacks=[self.tensorboard_callback]
                )

                self.model.save_weights(self.model_weights_path)

    def evaluate(self):
        logger.info("evaluating the model...")

        # TODO probably create 4th model without decoder_input_data for evaluation?
        # maybe not
        self.model.evaluate(
            [
                self.test_dataset["encoder_input_data"],
                self.test_dataset["decoder_input_data"]
            ],
            self.test_dataset["decoder_target_data"],
            batch_size=self.batch_size
        )

        if self.eval_translation:
            logger.info("Translating test dataset for BLEU evaluation...")
            path = self.test_dataset_path + "." + self.target_lang + ".translated"

            with open(path, "w", encoding="utf-8") as out_file:
                for seq in self.test_dataset["encoder_input_data"]:
                    decoded_sentence = self.decode_sequence(
                        seq, self.encoder_model, self.decoder_model,
                        self.test_dataset["y_ix_to_word"],
                        self.test_dataset["y_vocab_len"],
                        self.test_dataset["y_max_seq_len"]
                    )

                    out_file.write(decoded_sentence + "\n")

                    # TODO moses

    def translate(self, seq=None):
        """

        Translates either given sequence or random sequence from training source dataset to target language

        Args:
            seq: if given, sequence that will be translated from source to target language.

        """
        expected_seq = None

        if seq:
            encoded_seq = Translator.encode_text_to_input_seq(seq, self.training_dataset["x_word_to_ix"])
        else:
            # TODO what about bucketing
            encoder_input_data = self.training_dataset["encoder_input_data"][0]
            i = random.randint(0, len(encoder_input_data) - 1)
            encoded_seq = encoder_input_data[i]
            seq = " ".join(self.training_dataset["x_word_seq"][i])
            expected_seq = " ".join(self.training_dataset["y_word_seq"][i][1:-1])
            encoded_seq = encoded_seq.reshape((1, len(encoded_seq)))

        decoded_sentence = self.decode_sequence(
            encoded_seq, self.encoder_model, self.decoder_model,
            self.training_dataset["y_ix_to_word"],
            self.training_dataset["y_vocab_len"],
            50 # TODO y_max_seq_len
        )

        logger.info("Input sequence: {}".format(seq))
        logger.info("Expcected sentence: {}".format(expected_seq))
        logger.info("Translated sentence: {}".format(decoded_sentence))
