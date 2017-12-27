# coding: utf-8

import logging
import argparse
import os
import sys

# TODO how to properly log
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def bool_arg(arg):
    return True if arg == 'True' or arg is True else False


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.add_argument("--training_dataset", type=str, help="Path to the training set", required=True)
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path to the test set. Dataset are two files (one source one target language)."
                             + "Each line of a file is one sequence corresponding with the line of the second file.")
    parser.add_argument("--source_lang", type=str, help="Source language (dataset file extension)", required=True)
    parser.add_argument("--target_lang", type=str, help="Target language (dataset file extension)", required=True)
    parser.add_argument("--model_folder", type=str, default="model/", help="Path where the result model will be stored")
    parser.add_argument("--log_folder", type=str, default="logs/", help="Path where the result logs will be stored")
    parser.add_argument("--model_file", type=str, default="model_weights.h5",
                        help="Model file name. Either will be created or loaded.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="Number of layers in encoder")
    parser.add_argument("--num_decoder_layers", type=int, default=1, help="Number of layers in decoder")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of one batch")
    parser.add_argument("--num_units", type=int, default=256,
                        help="Size of each network layer")
    parser.add_argument("--optimizer", type=str, default="rmsprop", help="Keras optimizer name")
    parser.add_argument("--num_training_samples", type=int, default=-1,
                        help="How many samples to take from the training dataset, -1 for all of them")
    parser.add_argument("--num_test_samples", type=int, default=-1,
                        help="How many samples to take from the test dataset, -1 for all of them")
    parser.add_argument("--max_source_vocab_size", type=int, default=15000,
                        help="Maximum size of source vocabulary")
    parser.add_argument("--max_target_vocab_size", type=int, default=15000,
                        help="Maximum size of target vocabulary")
    parser.add_argument("--source_embedding_path", type=str, default=None,
                        help="Path to pretrained fastText embeddings file for source langauge")
    parser.add_argument("--target_embedding_path", type=str, default=None,
                        help="Path to pretrained fastText embeddings file for target language")
    parser.add_argument("--source_embedding_dim", type=int, default=300, help="Dimension of source embeddings")
    parser.add_argument("--target_embedding_dim", type=int, default=300, help="Dimension of target embeddings")
    parser.add_argument("--max_source_embedding_num", type=int, default=None,
                        help="how many first lines from embedding file should be loaded, None means all of them")
    parser.add_argument("--max_target_embedding_num", type=int, default=None,
                        help="how many first lines from embedding file should be loaded, None means all of them")
    parser.add_argument("--validation_split", type=float, default=0.0,
                        help="How big proportion of a development dataset should be used for validation during fiting")
    parser.add_argument("--bucketing", type=bool_arg, default=False,
                        help="Whether to bucket sequences according their size to optimize padding")
    parser.add_argument("--bucket_range", type=int, default=10,
                        help="Range of different sequence lenghts in one bucket")
    parser.add_argument("--use_fit_generator", type=bool_arg, default=False,
                        help="Prevent memory crash by only load part of the dataset at once each time when fitting")
    parser.add_argument("--reverse_input", type=bool_arg, default=True,
                        help="Whether to reverse source sequences (optimization for better learning)")
    parser.add_argument("--tokenize", type=bool_arg, default=True,
                        help="Whether to tokenize the sequences or not (they are already tokenizes e.g. using Moses tokenizer)")
    parser.add_argument("--clear", type=bool_arg, default=False,
                        help="Whether to delete old weights and logs before running")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--training_mode', action='store_true', default=False,
                            help="Trains the model on the training dataset and evaluate on testing dataset")
    mode_group.add_argument('--livetest_mode', action='store_true', default=False,
                            help="Loads trained model and lets user try translation in promt")


# TODO compare use_fit_generator speed True vs False

# python main.py --training_mode --training_dataset "data/anki_ces-eng" --test_dataset "data/OpenSubtitles2016-moses-10000.cs-en-tokenized.truecased.cleaned" --source_lang "cs" --target_lang "en" --num_units 100 --num_training_samples 100 --num_test_samples 100 --clear True --use_fit_generator False
# python main.py --training_mode --training_dataset "data/mySmallTest" --test_dataset "data/mySmallTest" --source_lang "cs" --target_lang "en" --epochs 1 --clear True
def main():
    parser = argparse.ArgumentParser(description='Arguments for the main.py that uses nmt module')
    add_arguments(parser)

    args, unparsed = parser.parse_known_args()

    if unparsed:
        logger.warning("some unexpected arguments: {}".format(unparsed))

    # to speed up loading of parser help
    # tensorflow takes quite some time to load
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'nmtPackage')))
    from nmt import Translator

    # this could be fix for the errors that are thrown sometimes
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    translator = Translator(
        source_embedding_dim=args.source_embedding_dim, target_embedding_dim=args.target_embedding_dim,
        source_embedding_path=args.source_embedding_path, target_embedding_path=args.target_embedding_path,
        max_source_embedding_num=args.max_source_embedding_num, max_target_embedding_num=args.max_target_embedding_num,
        epochs=args.epochs, source_lang=args.source_lang,
        num_units=args.num_units, optimizer=args.optimizer, use_fit_generator=args.use_fit_generator,
        log_folder=args.log_folder, max_source_vocab_size=args.max_source_vocab_size,
        max_target_vocab_size=args.max_target_vocab_size, model_file=args.model_file, model_folder=args.model_folder,
        num_training_samples=args.num_training_samples, num_test_samples=args.num_test_samples,
        reverse_input=args.reverse_input, target_lang=args.target_lang,
        test_dataset=args.test_dataset, training_dataset=args.training_dataset, validaton_split=args.validation_split,
        tokenize=args.tokenize, clear=args.clear
    )

    # TODO osamostatnit veci v modulu a vyndat je sem, z modulu udelat jen generic modul

    if args.training_mode:
        translator.fit(args.batch_size)
        evaluation = translator.evaluate(args.batch_size)

        print("model evaluation: {}".format(evaluation))

    elif args.livetest_mode:
        while True:
            seq = input("Enter sequence: ")
            translator.translate(seq)

    # TODO class for encoder/decoder


# autogenerate docs
# docs 	sphinx-apidoc -o docs nmt

if __name__ == "__main__":
    main()
