# coding: utf-8
# model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

import logging
import os
import argparse
import pickle
from time import time

import numpy as np
import nmt.utils as utils
from nmt import SpecialSymbols, Dataset, Vocabulary
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
    """

    Main class of the module, takes care of the datasets, fitting, evaluation and translating

    """

    def __init__(self, batch_size, bucketing, bucket_range, embedding_dim, embedding_path,
                 max_embedding_num, epochs, eval_translation,
                 source_lang, latent_dim,
                 log_folder, max_source_vocab_size, max_target_vocab_size, model_file, model_folder, num_samples,
                 reverse_input,
                 target_lang, test_dataset, training_dataset, validaton_split, clear):
        self.batch_size = batch_size
        self.bucketing = bucketing
        self.bucket_range = bucket_range
        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.max_embedding_num = max_embedding_num
        self.epochs = epochs
        self.eval_translation = eval_translation
        self.source_lang = source_lang
        self.latent_dim = latent_dim
        self.log_folder = log_folder
        self.max_source_vocab_size = max_source_vocab_size
        self.max_target_vocab_size = max_target_vocab_size
        # self.model_file = model_file
        self.model_folder = model_folder
        self.model_weights_path = "{}".format(os.path.join(model_folder, model_file))
        self.num_samples = num_samples
        self.reverse_input = reverse_input
        self.target_lang = target_lang
        self.test_dataset_path = test_dataset
        self.training_dataset_path = training_dataset
        self.validation_split = validaton_split
        self.clear = clear

        utils.prepare_folders([self.log_folder, self.model_folder], clear)

        self.training_dataset = Dataset(self.training_dataset_path, self.source_lang, self.target_lang,
                                        self.num_samples,
                                        True)  # TODO probably create parameter for it (tokenize), Moses tokenization will be used later on
        self.test_dataset = Dataset(self.test_dataset_path, self.source_lang, self.target_lang,
                                    self.num_samples,
                                    True)  # TODO probably create parameter for it (tokenize), Moses tokenization will be used later on

        logger.info("There are {} samples in datasets".format(self.training_dataset.num_samples))

        self.source_vocab = Vocabulary(self.training_dataset.x_word_seq, self.max_source_vocab_size)
        self.target_vocab = Vocabulary(self.training_dataset.y_word_seq, self.max_target_vocab_size)

        self.training_data = self._prepare_training_data()
        self.test_data = self._prepare_testing_data()

        self.embedding_weights = None
        if not os.path.isfile(self.model_weights_path) and self.embedding_path:
            # load pretrained embeddings
            self.embedding_weights = utils.load_embedding_weights(self.embedding_path,
                                                                  self.source_vocab.ix_to_word,
                                                                  limit=self.max_embedding_num)

        self.model, self.encoder_model, self.decoder_model = self._define_models()

        self.model.summary()

        # logging for tensorboard
        self.tensorboard_callback = TensorBoard(log_dir="{}".format(os.path.join(self.log_folder, str(time()))),
                                                write_graph=False)

        logger.info("compiling model...")
        # Run training
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           metrics=['acc'])

        if os.path.isfile(self.model_weights_path):
            logger.info("Loading model weights from file..")
            self.model.load_weights(self.model_weights_path)

    def _prepare_training_data(self):
        if self.bucketing:
            encoder_input_data = []
            decoder_input_data = []
            decoder_target_data = []

            buckets = utils.split_to_buckets(self.training_dataset.x_word_seq, self.training_dataset.y_word_seq,
                                             self.bucket_range,
                                             self.training_dataset.x_max_seq_len, self.training_dataset.y_max_seq_len)

            for ix, bucket in buckets.items():
                enc_in, dec_in, dec_tar = self._encode_sequences(
                    bucket["x_word_seq"], bucket["y_word_seq"],
                    bucket["x_max_seq_len"], bucket["y_max_seq_len"],
                    self.source_vocab.word_to_ix, self.target_vocab.word_to_ix
                )

                encoder_input_data.append(enc_in)
                decoder_input_data.append(dec_in)
                decoder_target_data.append(dec_tar)
        else:
            encoder_input_data, decoder_input_data, decoder_target_data = self._encode_sequences(
                self.training_dataset.x_word_seq, self.training_dataset.y_word_seq,
                self.training_dataset.x_max_seq_len, self.training_dataset.y_max_seq_len,
                self.source_vocab.word_to_ix, self.target_vocab.word_to_ix
            )

            encoder_input_data = [encoder_input_data]
            decoder_input_data = [decoder_input_data]
            decoder_target_data = [decoder_target_data]

        return {
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def _prepare_testing_data(self):
        """
        # vocabularies of test dataset has to be the same as of training set
        # otherwise embeddings would not correspond are use OOV
        # and y one hot encodings wouldnt correspond either

        Returns:

        """

        encoder_input_data, decoder_input_data, decoder_target_data = self._encode_sequences(
            self.test_dataset.x_word_seq, self.test_dataset.y_word_seq,
            self.test_dataset.x_max_seq_len, self.test_dataset.y_max_seq_len,
            self.source_vocab.word_to_ix, self.target_vocab.word_to_ix
        )

        return {
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def _encode_sequences(self, x_word_seq, y_word_seq,
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
                    encoder_input_data[i][t] = SpecialSymbols.UNK_ID

        # encode target sentences to one hot encoding
        for i, seq in enumerate(y_word_seq):
            for t, word in enumerate(seq):
                if word in y_word_to_ix:
                    index = y_word_to_ix[word]
                else:
                    index = SpecialSymbols.UNK_ID
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t][index] = 1

                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, index] = 1

        return encoder_input_data, decoder_input_data, decoder_target_data

    def _define_models(self):
        logger.info("Creating models...")
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))

        if self.embedding_weights is not None:
            self.embedding_weights = [self.embedding_weights]  # Embedding layer wantes list as parameter
        # TODO trainable False or True?
        # TODO according to https://keras.io/layers/embeddings/
        # input dim should be +1 when used with mask_zero..is it correctly set here?
        embedding = Embedding(self.source_vocab.vocab_len, self.embedding_dim,
                              weights=self.embedding_weights, mask_zero=True)
        embedding_outputs = embedding(encoder_inputs)

        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(embedding_outputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.target_vocab.vocab_len))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.target_vocab.vocab_len, activation='softmax')
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
        target_seq[0, 0, SpecialSymbols.GO_ID] = 1.

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
            if sampled_word == SpecialSymbols.EOS:
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
                ix = SpecialSymbols.UNK_ID
            x[0][i] = ix

        return x

    def fit(self):
        """

        fits the model, according to the parameters passed in constructor

        """
        logger.info("fitting the model...")
        for i in range(self.epochs):
            logger.info("Epoch {}".format(i + 1))

            for j in range(len(self.training_data["encoder_input_data"])):
                if self.bucketing:
                    logger.info("Bucket {}".format(j))

                self.model.fit(
                    [
                        self.training_data["encoder_input_data"][j],
                        self.training_data["decoder_input_data"][j]
                    ],
                    self.training_data["decoder_target_data"][j],
                    batch_size=self.batch_size,
                    epochs=1,
                    validation_split=self.validation_split,
                    callbacks=[self.tensorboard_callback]
                )

                self.model.save_weights(self.model_weights_path)

    def evaluate(self):
        """

        performs evaluation on test dataset along with generating translations
        and calculating BLEU score for the dataset

        Returns: Keras model.evaluate values

        """
        logger.info("evaluating the model...")

        # TODO probably create 4th model without decoder_input_data for evaluation?
        # maybe not
        eval_values = self.model.evaluate(
            [
                self.test_data["encoder_input_data"],
                self.test_data["decoder_input_data"]
            ],
            self.test_data["decoder_target_data"],
            batch_size=self.batch_size
        )

        if self.eval_translation:
            logger.info("Translating test dataset for BLEU evaluation...")
            path_original = self.test_dataset_path + "." + self.target_lang
            path = path_original + ".translated"

            with open(path, "w", encoding="utf-8") as out_file:
                for seq in self.test_data["encoder_input_data"]:
                    decoded_sentence = self.decode_sequence(
                        seq, self.encoder_model, self.decoder_model,
                        self.target_vocab.ix_to_word,
                        self.target_vocab.vocab_len,
                        self.test_dataset.y_max_seq_len
                    )

                    out_file.write(decoded_sentence + "\n")

            utils.get_bleu(path_original, path)

        return eval_values

    def translate(self, seq=None):
        """

        Translates either given sequence or random sequence from training source dataset to target language

        Args:
            seq: if given, sequence that will be translated from source to target language.

        """
        expected_seq = None

        if seq:
            encoded_seq = Translator.encode_text_to_input_seq(seq, self.source_vocab.word_to_ix)
        else:
            # TODO what about bucketing
            encoder_input_data = self.training_data["encoder_input_data"][0]
            i = random.randint(0, len(encoder_input_data) - 1)
            encoded_seq = encoder_input_data[i]
            seq = " ".join(self.training_dataset.x_word_seq[i])
            expected_seq = " ".join(self.training_dataset.y_word_seq[i][1:-1])
            encoded_seq = encoded_seq.reshape((1, len(encoded_seq)))

        decoded_sentence = self.decode_sequence(
            encoded_seq, self.encoder_model, self.decoder_model,
            self.target_vocab.ix_to_word,
            self.target_vocab.vocab_len,
            50  # TODO y_max_seq_len
        )

        logger.info("Input sequence: {}".format(seq))
        logger.info("Expcected sentence: {}".format(expected_seq))
        logger.info("Translated sentence: {}".format(decoded_sentence))
