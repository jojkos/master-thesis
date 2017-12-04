# coding: utf-8
# model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

import logging
import argparse
from nmt.translator import Translator

# TODO how to properly log
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.add_argument("--training_dataset", type=str, help="Path to the training set", required=True)
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path to the test set. Dataset are two files (one source one target language)."
                             + "Each line of a file is one sequence corresponding with the line of the second file.")
    parser.add_argument("--in_lang", type=str, help="Source language (dataset file extension)", required=True)
    parser.add_argument("--target_lang", type=str, help="Target language (dataset file extension)", required=True)
    parser.add_argument("--model_folder", type=str, default="model/", help="Path where the result model will be stored")
    parser.add_argument("--log_folder", type=str, default="logs/", help="Path where the result logs will be stored")
    parser.add_argument("--model_file", type=str, default="model_weights.h5", help="Model file name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of one batch")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="TODO maybe num_units instead? size of each network layer")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="How many samples to take from the dataset, 0 for all of them")
    parser.add_argument("--max_in_vocab_size", type=int, default=15000,
                        help="Maximum size of source vocabulary, 0 for unlimited")
    parser.add_argument("--max_out_vocab_size", type=int, default=15000,
                        help="Maximum size of target vocabulary, 0 for unlimited")
    parser.add_argument("--embedding_path", type=str, default=None, help="Path to pretrained fastText embeddings file")
    parser.add_argument("--embedding_dim", type=int, default=300, help="Dimension of embeddings")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="How big proportion of a development dataset should be used for validation during fiting")
    parser.add_argument("--bucketing", type=bool, default=False,
                        help="Whether to bucket sequences according their size to optimize padding")
    parser.add_argument("--bucket_range", type=int, default=10,
                        help="Range of different sequence lenghts in one bucket")
    parser.add_argument("--reverse_input", type=bool, default=True,
                        help="Whether to reverse source sequences (optimization for better learning)")
    parser.add_argument("--eval_translation", type=bool, default=True,
                        help="Whether to generate translation for the test dataset and then compute the BLEU score")


# python main.py --training_dataset "data/anki_ces-eng" --test_dataset "data/news-commentary-v9.cs-en" --in_lang "cs" --target_lang "en" --latent_dim 100 --num_samples 100
def main():
    parser = argparse.ArgumentParser(description='Arguments for the Translator class')
    add_arguments(parser)

    args, unparsed = parser.parse_known_args()

    if unparsed:
        logger.warning("some unexpected arguments: {}".format(unparsed))

    translator = Translator(
        batch_size=args.batch_size, bucketing=args.bucketing, bucket_range=args.bucket_range,
        embedding_dim=args.embedding_dim, embedding_path=args.embedding_path, epochs=args.epochs,
        eval_translation=args.eval_translation, in_lang=args.in_lang,
        latent_dim=args.latent_dim, log_folder=args.log_folder, max_in_vocab_size=args.max_in_vocab_size,
        max_out_vocab_size=args.max_out_vocab_size, model_file=args.model_file, model_folder=args.model_folder,
        num_samples=args.num_samples, reverse_input=args.reverse_input, target_lang=args.target_lang,
        test_dataset=args.test_dataset, training_dataset=args.training_dataset, validaton_split=args.validation_split
    )
    translator.fit()
    translator.evaluate()

    translator.translate()
    translator.translate("kočka chodí dírou")

    # TODO refactor methods in translator to use self instead of parameters
    # TODO class for dataset
    # TODO class for encoder/decoder


if __name__ == "__main__":
    main()
