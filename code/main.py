# coding: utf-8

import logging
import argparse
from nmt.translator import Translator

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
    parser.add_argument("--embedding_path", type=str, default=None, help="Path to pretrained fastText embeddings file")
    parser.add_argument("--embedding_dim", type=int, default=300, help="Dimension of embeddings")
    parser.add_argument("--max_embedding_num", type=int, default=None,
                        help="how many first lines from embedding file should be loaded, None means all of them(irony)")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="How big proportion of a development dataset should be used for validation during fiting")
    parser.add_argument("--bucketing", type=bool_arg, default=False,
                        help="Whether to bucket sequences according their size to optimize padding")
    parser.add_argument("--bucket_range", type=int, default=10,
                        help="Range of different sequence lenghts in one bucket")
    parser.add_argument("--use_fit_generator", type=bool_arg, default=True,
                        help="Prevent memory crash by only load part of the dataset at once each time when fitting")
    parser.add_argument("--reverse_input", type=bool_arg, default=True,
                        help="Whether to reverse source sequences (optimization for better learning)")
    parser.add_argument("--tokenize", type=bool_arg, default=True,
                        help="Whether to tokenize the sequences or not (they are already tokenizes e.g. using Moses tokenizer)")
    parser.add_argument("--clear", type=bool_arg, default=False,
                        help="Whether to delete old weights and logs before running")


# TODO compare use_fit_generator speed True vs False

# python main.py --training_dataset "data/anki_ces-eng" --test_dataset "data/OpenSubtitles2016-moses-10000.cs-en-tokenized.truecased.cleaned" --source_lang "cs" --target_lang "en" --num_units 100 --num_training_samples 100 --num_test_samples 100 --clear True --use_fit_generator True
def main():
    parser = argparse.ArgumentParser(description='Arguments for the Translator class')
    add_arguments(parser)

    args, unparsed = parser.parse_known_args()

    if unparsed:
        logger.warning("some unexpected arguments: {}".format(unparsed))

    translator = Translator(
        batch_size=args.batch_size, bucketing=args.bucketing, bucket_range=args.bucket_range,
        embedding_dim=args.embedding_dim, embedding_path=args.embedding_path,
        max_embedding_num=args.max_embedding_num, epochs=args.epochs, source_lang=args.source_lang,
        num_units=args.num_units, optimizer=args.optimizer, use_fit_generator=args.use_fit_generator,
        log_folder=args.log_folder, max_source_vocab_size=args.max_source_vocab_size,
        max_target_vocab_size=args.max_target_vocab_size, model_file=args.model_file, model_folder=args.model_folder,
        num_training_samples=args.num_training_samples, num_test_samples=args.num_test_samples,
        reverse_input=args.reverse_input, target_lang=args.target_lang,
        test_dataset=args.test_dataset, training_dataset=args.training_dataset, validaton_split=args.validation_split,
        tokenize=args.tokenize, clear=args.clear
    )
    translator.fit()
    evaluation = translator.evaluate()

    logger.info("model evaluation: {}".format(evaluation))

    # translator.translate()
    translator.translate("kočka chodí dírou")

    # TODO class for encoder/decoder


# autogenerate docs
# docs 	sphinx-apidoc -o docs nmt

if __name__ == "__main__":
    main()
