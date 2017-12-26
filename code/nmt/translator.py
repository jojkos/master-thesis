# coding: utf-8
# model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

import logging
import os
import math
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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Translator(object):
    """

    Main class of the module, takes care of the datasets, fitting, evaluation and translating

    """

    def __init__(self, batch_size, bucketing, bucket_range,
                 source_embedding_dim, target_embedding_dim, source_embedding_path, target_embedding_path,
                 max_source_embedding_num, max_target_embedding_num, epochs, use_fit_generator,
                 source_lang, num_units, optimizer,
                 log_folder, max_source_vocab_size, max_target_vocab_size, model_file, model_folder,
                 reverse_input, target_lang, test_dataset, training_dataset, validaton_split, clear,
                 tokenize, num_training_samples=-1, num_test_samples=-1):
        """

        Args:
            batch_size (int): Size of one batch
            bucketing (bool): Whether to bucket sequences according their size to optimize padding
            bucket_range (int): Range of different sequence lenghts in one bucket
            source_embedding_dim (int): Dimension of embeddings
            target_embedding_dim (int): Dimension of embeddings
            target_embedding_path (str): Path to pretrained fastText embeddings file
            max_source_embedding_num (int): how many first lines from embedding file should be loaded, None means all of them
            epochs (int): Number of epochs
            source_lang (str): Source language (dataset file extension)
            num_units (str): Size of each network layer
            optimizer (str): Keras optimizer name
            log_folder (str): Path where the result logs will be stored
            max_source_vocab_size (int): Maximum size of source vocabulary
            max_target_vocab_size (int): Maximum size of target vocabulary
            model_file (str): Model file name. Either will be created or loaded.
            model_folder (str): Path where the result model will be stored
            num_training_samples (int, optional): How many samples to take from the training dataset, -1 for all of them (default)
            num_test_samples (int, optional): How many samples to take from the test dataset, -1 for all of them (default)
            reverse_input (bool): Whether to reverse source sequences (optimization for better learning)
            target_lang (str): Target language (dataset file extension)
            test_dataset (str): Path to the test set. Dataset are two files (one source one target language)
            training_dataset (str): Path to the training set
            validaton_split (float): How big proportion of a development dataset should be used for validation during fiting
            clear (bool): Whether to delete old weights and logs before running
            tokenize (bool): Whether to tokenize the sequences or not (they are already tokenizes e.g. using Moses tokenizer)
            use_fit_generator (bool): Prevent memory crash by only load part of the dataset at once each time when fitting"
        """

        self.batch_size = batch_size
        self.bucketing = bucketing
        self.bucket_range = bucket_range
        self.source_embedding_dim = source_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.source_embedding_path = source_embedding_path
        self.target_embedding_path = target_embedding_path
        self.max_source_embedding_num = max_source_embedding_num
        self.max_target_embedding_num = max_target_embedding_num
        self.epochs = epochs
        self.use_fit_generator = use_fit_generator
        self.source_lang = source_lang
        self.num_units = num_units
        self.optimizer = optimizer
        self.log_folder = log_folder
        self.max_source_vocab_size = max_source_vocab_size
        self.max_target_vocab_size = max_target_vocab_size
        self.model_folder = model_folder
        self.model_weights_path = "{}".format(os.path.join(model_folder, model_file))
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples
        self.reverse_input = reverse_input
        self.target_lang = target_lang
        self.test_dataset_path = test_dataset
        self.training_dataset_path = training_dataset
        self.validation_split = validaton_split
        self.clear = clear
        self.tokenize = tokenize

        utils.prepare_folders([self.log_folder, self.model_folder], clear)

        self.training_dataset = Dataset(self.training_dataset_path, self.source_lang, self.target_lang,
                                        self.num_training_samples,
                                        self.tokenize)
        self.test_dataset = Dataset(self.test_dataset_path, self.source_lang, self.target_lang,
                                    self.num_test_samples,
                                    self.tokenize)

        logger.info("There are {} samples in training dataset".format(self.training_dataset.num_samples))
        logger.info("There are {} samples in test dataset".format(self.test_dataset.num_samples))

        self.source_vocab = Vocabulary(self.training_dataset.x_word_seq, self.max_source_vocab_size, False)
        self.target_vocab = Vocabulary(self.training_dataset.y_word_seq, self.max_target_vocab_size, True)

        logger.info("Source vocabulary has {} symbols".format(self.source_vocab.vocab_len))
        logger.info("Target vocabulary has {} symbols".format(self.target_vocab.vocab_len))

        self.source_embedding_weights = None
        if not os.path.isfile(self.model_weights_path) and self.source_embedding_path:
            # load pretrained embeddings
            self.source_embedding_weights = utils.load_embedding_weights(self.source_embedding_path,
                                                                         self.source_vocab.ix_to_word,
                                                                         limit=self.max_source_embedding_num)

        self.target_embedding_weights = None
        if not os.path.isfile(self.model_weights_path) and self.target_embedding_path:
            # load pretrained embeddings
            self.source_embedding_weights = utils.load_embedding_weights(self.target_embedding_path,
                                                                         self.target_vocab.ix_to_word,
                                                                         limit=self.max_target_embedding_num)

        self.model, self.encoder_model, self.decoder_model = self._define_models()

        self.model.summary()

        # TODO uncomment
        # logging for tensorboard
        # self.tensorboard_callback = TensorBoard(log_dir="{}".format(os.path.join(self.log_folder, str(time()))),
        #                                         write_graph=False)  # quite SLOW LINE

        logger.info("compiling model...")
        # Run training
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                           metrics=['acc'])

        if os.path.isfile(self.model_weights_path):
            logger.info("Loading model weights from file..")
            self.model.load_weights(self.model_weights_path)

    def _get_training_data(self, from_index=0, to_index=None):
        """

        Returns: dict with encoder_input_data, decoder_input_data and decoder_target_data of whole dataset size

        """
        if self.bucketing:
            encoder_input_data = []
            decoder_input_data = []
            decoder_target_data = []

            buckets = utils.split_to_buckets(self.training_dataset.x_word_seq[from_index: to_index],
                                             self.training_dataset.y_word_seq[from_index: to_index],
                                             self.bucket_range,
                                             self.training_dataset.x_max_seq_len,
                                             self.training_dataset.y_max_seq_len)

            for ix, bucket in buckets.items():
                enc_in, dec_in, dec_tar = Translator.encode_sequences(
                    bucket["x_word_seq"], bucket["y_word_seq"],
                    bucket["x_max_seq_len"], bucket["y_max_seq_len"],
                    self.source_vocab.word_to_ix, self.target_vocab.word_to_ix, self.reverse_input
                )

                encoder_input_data.append(enc_in)
                decoder_input_data.append(dec_in)
                decoder_target_data.append(dec_tar)
        else:
            encoder_input_data, decoder_input_data, decoder_target_data = Translator.encode_sequences(
                self.training_dataset.x_word_seq[from_index: to_index],
                self.training_dataset.y_word_seq[from_index: to_index],
                self.training_dataset.x_max_seq_len, self.training_dataset.y_max_seq_len,
                self.source_vocab.word_to_ix, self.target_vocab.word_to_ix, self.reverse_input
            )

            encoder_input_data = [encoder_input_data]
            decoder_input_data = [decoder_input_data]
            decoder_target_data = [decoder_target_data]

        return {
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def _training_data_gen(self, infinite=True):
        """

        Args:
            infinite: whether to yield data infinitely or stop after one walkthrough the dataset

        Returns: dict with encoder_input_data, decoder_input_data and decoder_target_data of self.batch_size size

        """
        # TODO what about shuffling?
        # maybe use keras.sequence instead of generator?
        # https://keras.io/utils/#sequence
        # https://stackoverflow.com/questions/46570172/how-to-fit-generator-in-keras
        # https://github.com/keras-team/keras/issues/2389 probably don't need to use sequence but have to shuffle data HERE manually

        i = 0
        once_through = False

        while infinite or not once_through:
            training_data = self._get_training_data(i, i + self.batch_size)

            if self.bucketing:
                yield training_data
            else:
                yield (
                    [training_data["encoder_input_data"], training_data["decoder_input_data"]],
                    training_data["decoder_target_data"]
                )

            i += self.batch_size

            if i >= self.training_dataset.num_samples:
                once_through = True
                i = 0

    def _get_test_data(self, from_index=0, to_index=None):
        encoder_input_data, decoder_input_data, decoder_target_data = Translator.encode_sequences(
            self.test_dataset.x_word_seq[from_index: to_index],
            self.test_dataset.y_word_seq[from_index: to_index],
            self.test_dataset.x_max_seq_len, self.test_dataset.y_max_seq_len,
            self.source_vocab.word_to_ix, self.target_vocab.word_to_ix, self.reverse_input
        )

        return {
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def _test_data_gen(self, infinite=True):
        """
        # vocabularies of test dataset has to be the same as of training set
        # otherwise embeddings would not correspond are use OOV
        # and y one hot encodings wouldnt correspond either

        Args:
            infinite: whether to run infinitely or just do one loop over the dataset

        Yields: x inputs, y inputs

        """

        i = 0
        once_through = False

        while infinite or not once_through:
            test_data = self._get_test_data(i, i + self.batch_size)

            yield (
                [test_data["encoder_input_data"], test_data["decoder_input_data"]],
                test_data["decoder_target_data"]
            )

            i += self.batch_size

            if i >= self.test_dataset.num_samples:
                once_through = True
                i = 0

    @staticmethod
    def encode_sequences(x_word_seq, y_word_seq,
                         x_max_seq_len, y_max_seq_len,
                         x_word_to_ix, y_word_to_ix, reverse_input):
        """
        Take word sequences and convert them so that the model can be fit with them.
        Input words are just converted to integer index
        Target words are encoded to one hot vectors of target vocabulary length
        """
        y_vocab_len = len(y_word_to_ix)

        # if we try to allocate memory for whole dataset (even for not a big one), Memory Error is raised
        # always encode only a part of the dataset
        encoder_input_data = np.zeros(
            (len(x_word_seq), x_max_seq_len), dtype='float32')
        decoder_input_data = np.zeros(
            (len(x_word_seq), y_max_seq_len), dtype='float32')
        decoder_target_data = np.zeros(
            (len(x_word_seq), y_max_seq_len, y_vocab_len),
            dtype='float32')

        # prepare source sentences for embedding layer (encode to indexes)
        for i, seq in enumerate(x_word_seq):
            if reverse_input:  # for better results according to paper Sequence to seq...
                seq = seq[::-1]
            for t, word in enumerate(seq):
                if word in x_word_to_ix:
                    encoder_input_data[i, t] = x_word_to_ix[word]
                else:
                    encoder_input_data[i, t] = SpecialSymbols.UNK_IX

        # encode target sentences to one hot encoding
        for i, seq in enumerate(y_word_seq):
            for t, word in enumerate(seq):
                if word in y_word_to_ix:
                    index = y_word_to_ix[word]
                else:
                    index = SpecialSymbols.UNK_IX
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t] = index

                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, index] = 1

        return encoder_input_data, decoder_input_data, decoder_target_data

    def _define_models(self):
        # model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        logger.info("Creating models...")
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))

        if self.source_embedding_weights is not None:
            self.source_embedding_weights = [self.source_embedding_weights]  # Embedding layer wantes list as parameter
        # TODO trainable False or True?
        # according to https://keras.io/layers/embeddings/
        # input dim should be +1 when used with mask_zero..is it correctly set here?
        # i think that input dim is already +1 because padding symbol is part of the vocabulary
        source_embeddings = Embedding(self.source_vocab.vocab_len, self.source_embedding_dim,
                                      weights=self.source_embedding_weights, mask_zero=True, trainable=True)
        source_embedding_outputs = source_embeddings(encoder_inputs)

        encoder = LSTM(self.num_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(source_embedding_outputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        if self.target_embedding_weights is not None:
            self.target_embedding_weights = [self.target_embedding_weights]  # Embedding layer wantes list as parameter
        target_embeddings = Embedding(self.target_vocab.vocab_len, self.target_embedding_dim,
                                      weights=self.target_embedding_weights, mask_zero=True, trainable=True)
        target_embedding_outputs = target_embeddings(decoder_inputs)

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.num_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(target_embedding_outputs,
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

        decoder_state_input_h = Input(shape=(self.num_units,))
        decoder_state_input_c = Input(shape=(self.num_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            target_embedding_outputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = SpecialSymbols.GO_IX

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1). # TODO ? can the batch size be bigger?
        decoded_sentence = ""
        while True:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.target_vocab.ix_to_word[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_word == SpecialSymbols.EOS:
                break

            decoded_sentence += sampled_word + " "
            decoded_len = len(decoded_sentence.strip().split(" "))

            if decoded_len > self.training_dataset.y_max_seq_len \
                    and decoded_len > self.test_dataset.y_max_seq_len:  # TODO maybe change to arbitrary long?
                break

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

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
                ix = SpecialSymbols.UNK_IX
            x[0][i] = ix

        return x

    def get_gen_steps(self, dataset):
        """

        Returns how many steps are needed for the generator to go through the whole dataset with the self.batch_size

        Args:
            dataset: dataset that is beeing proccessed

        Returns: number of steps for the generatorto go through whole dataset

        """
        return math.ceil(dataset.num_samples / self.batch_size)

    def fit(self):
        """

        fits the model, according to the parameters passed in constructor

        """
        logger.info("fitting the model...")
        for i in range(self.epochs):
            logger.info("Epoch {}".format(i + 1))

            if self.use_fit_generator:
                # to prevent memory error, only loads parts of dataset at once
                if self.bucketing:
                    for training_data in self._training_data_gen(infinite=False):

                        for j in range(len(training_data["encoder_input_data"])):
                            logger.info(
                                "Bucket {} size of {}".format(j, len(training_data["encoder_input_data"][j][0])))

                            self.model.fit(
                                [
                                    training_data["encoder_input_data"][j],
                                    training_data["decoder_input_data"][j]
                                ],
                                training_data["decoder_target_data"][j],
                                batch_size=self.batch_size,
                                epochs=1,
                                validation_split=self.validation_split,
                                # callbacks=[self.tensorboard_callback]
                            )
                else:
                    steps = self.get_gen_steps(self.training_dataset)
                    logger.info("traning generator will make {} steps".format(steps))
                    # TODO why is there no validation split
                    self.model.fit_generator(self._training_data_gen(),
                                             steps_per_epoch=steps,
                                             epochs=1,
                                             # callbacks=[self.tensorboard_callback]
                                             )
                    # validation_data=self._get_all_test_data()
            else:
                training_data = self._get_training_data()

                for j in range(len(training_data["encoder_input_data"])):
                    if self.bucketing:
                        logger.info(
                            "Bucket {} size of {}".format(j, len(training_data["encoder_input_data"][j][0])))

                    self.model.fit(
                        [
                            training_data["encoder_input_data"][j],
                            training_data["decoder_input_data"][j]
                        ],
                        training_data["decoder_target_data"][j],
                        batch_size=self.batch_size,
                        epochs=1,
                        validation_split=self.validation_split,
                        # callbacks=[self.tensorboard_callback]
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

        steps = self.get_gen_steps(self.test_dataset)
        logger.info("evaluation generator will make {} steps".format(steps))

        # test_data_gen gets called more then steps times,
        # probably because of the workers caching the values for optimization
        eval_data = self._test_data_gen()  # cannot be generator if want to use histograms in tensorboard callback
        eval_values = self.model.evaluate_generator(eval_data,
                                                    steps=steps)

        logger.info("Translating test dataset for BLEU evaluation...")
        path_original = self.test_dataset_path + "." + self.target_lang
        path = path_original + ".translated"

        step = 1
        with open(path, "w", encoding="utf-8") as out_file:
            for inputs, targets in self._test_data_gen(infinite=False):
                print("\rstep {} out of {}".format(step, steps), end="", flush=True)
                step += 1
                encoder_input_data = inputs[0]
                for i in range(len(encoder_input_data)):
                    # we need to keep the item in array ([i: i + 1])
                    decoded_sentence = self.decode_sequence(encoder_input_data[i: i + 1])

                    out_file.write(decoded_sentence + "\n")
        print("", end="\n")
        utils.get_bleu(path_original, path)

        return eval_values

    def translate(self, seq=None, expected_seq=None):
        """

        Translates given sequence

        Args:
            seq: sequence that will be translated from source to target language.
            expected_seq: optional, expected result of translation

        """

        encoded_seq = Translator.encode_text_to_input_seq(seq, self.source_vocab.word_to_ix)
        # else:
        #     encoder_input_data = self.training_data["encoder_input_data"][0]
        #     i = random.randint(0, len(encoder_input_data) - 1)
        #     encoded_seq = encoder_input_data[i]
        #     seq = " ".join(self.training_dataset.x_word_seq[i])
        #     expected_seq = " ".join(self.training_dataset.y_word_seq[i][1:-1])
        #     encoded_seq = encoded_seq.reshape((1, len(encoded_seq)))

        decoded_sentence = self.decode_sequence(encoded_seq)

        logger.info("Input sequence: {}".format(seq))
        logger.info("Expcected sentence: {}".format(expected_seq))
        print("Translated sentence: {}".format(decoded_sentence))
