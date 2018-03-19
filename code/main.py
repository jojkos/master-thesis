# coding: utf-8

import logging
import argparse
import os
import subprocess
import sys
import random

# seed to keep the results same every time
random.seed(0)

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
    parser.add_argument("--initial_epoch", type=int, default=0, help="Epoch number from which to start")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="Number of layers in encoder")
    parser.add_argument("--num_decoder_layers", type=int, default=1, help="Number of layers in decoder")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of one batch")
    parser.add_argument("--beam_size", type=int, default=1, help="Size of a beam for beam search decoding")
    parser.add_argument("--num_units", type=int, default=256,
                        help="Size of each network layer")
    parser.add_argument("--early_stopping_patience", type=int, default=-1,
                        help="How many epochs should model train after loss/val_loss decreases. -1 means that early stopping won't be used.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads for tensorflow configuration")
    parser.add_argument("--optimizer", type=str, default="rmsprop", help="Keras optimizer name")
    parser.add_argument("--dropout", type=int, default=0.2, help="Dropout size")
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
    parser.add_argument("--bucketing", type=bool_arg, default=False,
                        help="Whether to bucket sequences according their size to optimize padding")
    parser.add_argument("--bucket_range", type=int, default=3,
                        help="Range of different sequence lenghts in one bucket")
    parser.add_argument("--use_fit_generator", type=bool_arg, default=False,
                        help="Prevent memory crash by only load part of the dataset at once each time when fitting")
    parser.add_argument("--reverse_input", type=bool_arg, default=False,
                        help="Whether to reverse source sequences (optimization for better learning)")
    parser.add_argument("--tokenize", type=bool_arg, default=False,
                        help="Whether to tokenize the sequences or not (they are already tokenizes e.g. using Moses tokenizer)")
    parser.add_argument("--clear", type=bool_arg, default=False,
                        help="Whether to delete old weights and logs before running")
    parser.add_argument("--find_gpu", type=bool_arg, default=False,
                        help="Find and assign empty gpu to env var CUDA_VISIVLE_DEVICES. Otherwise use anything.")

    parser.add_argument('--train', action='store_true', default=False,
                        help="Trains the model on the training dataset")
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help="Evaluate on the test dataset")
    parser.add_argument('--livetest', action='store_true', default=False,
                        help="Loads trained model and lets user try translation in promt")


def set_gpu():
    free_gpu = subprocess.check_output(
        'nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"',
        shell=True)

    if len(free_gpu) == 0:
        logger.error('No free GPU available!')
        sys.exit(1)
    else:
        free_gpu = free_gpu.decode().strip()
        logger.info("Picked free GPU: {}".format(free_gpu))

    os.environ['CUDA_VISIBLE_DEVICES'] = free_gpu

    return free_gpu


# python main.py --train --training_dataset "data/anki_ces-eng" --test_dataset "data/OpenSubtitles2016-moses-10000.cs-en-tokenized.truecased.cleaned" --source_lang "cs" --target_lang "en" --num_units 100 --num_training_samples 100 --num_test_samples 100 --clear True --use_fit_generator False
# python main.py --train --training_dataset "data/mySmallTest" --test_dataset "data/mySmallTest" --source_lang "cs" --target_lang "en" --epochs 5 --log_folder "logs/smallTest"

# SMT pousteni
# python main.py --train --training_dataset "G:\Clouds\DPbigFiles\WMT17\newsCommentary\news-commentary-v12.cs-en-tokenized.truecased.cleaned" --test_dataset "G:\Clouds\DPbigFiles\WMT17\testSet\newstest2017-csen-tokenized.truecased.cleaned" --source_lang "cs" --target_lang "en" --model_folder "G:\Clouds\DPbigFiles\WMT17\newsCommentary" --model_file "newsCommentarySmtModel.h5" --batch_size 64 --num_units 256 --optimizer "rmsprop" --max_source_vocab_size 10000 --max_target_vocab_size 10000 --source_embedding_path "G:\Clouds\DPbigFiles\facebookVectors\facebookPretrained-wiki.cs.vec" --target_embedding_path "G:\Clouds\DPbigFiles\facebookVectors\facebookPretrained-wiki.en.vec"
def main():
    parser = argparse.ArgumentParser(description='Arguments for the main.py that uses nmt package')
    add_arguments(parser)

    args, unparsed = parser.parse_known_args()

    if not (args.train or args.evaluate or args.livetest):
        parser.error('At least one action requested, add -train, -evaluate or livetest')

    if unparsed:
        logger.warning("some unexpected arguments: {}".format(unparsed))

    if args.find_gpu:
        device = "/gpu:{}".format(set_gpu())
    else:
        device = "/gpu:0"

    # to speed up loading of parser help
    # tensorflow takes quite some time to load
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'nmtPackage')))
    from nmt import Translator, utils
    import tensorflow as tf

    with tf.device(device):
        translator = Translator(
            source_embedding_dim=args.source_embedding_dim, target_embedding_dim=args.target_embedding_dim,
            source_embedding_path=args.source_embedding_path, target_embedding_path=args.target_embedding_path,
            max_source_embedding_num=args.max_source_embedding_num,
            max_target_embedding_num=args.max_target_embedding_num,
            num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
            source_lang=args.source_lang, dropout=args.dropout,
            num_units=args.num_units, num_threads=args.num_threads, optimizer=args.optimizer,
            log_folder=args.log_folder, max_source_vocab_size=args.max_source_vocab_size,
            max_target_vocab_size=args.max_target_vocab_size, model_file=args.model_file,
            model_folder=args.model_folder,
            num_training_samples=args.num_training_samples, num_test_samples=args.num_test_samples,
            reverse_input=args.reverse_input, target_lang=args.target_lang,
            test_dataset=args.test_dataset, training_dataset=args.training_dataset,
            tokenize=args.tokenize, clear=args.clear
        )

        # TODO osamostatnit veci v modulu a vyndat je sem, z modulu udelat jen generic modul

        if args.train:
            translator.fit(epochs=args.epochs, initial_epoch=args.initial_epoch,
                           batch_size=args.batch_size, use_fit_generator=args.use_fit_generator,
                           bucketing=args.bucketing, bucket_range=args.bucket_range,
                           early_stopping_patience=args.early_stopping_patience)

        if args.evaluate:
            translator.translate_test_data(args.batch_size, args.beam_size)

            # remove bpe subwords before bleu scoring
            utils.restore_subwords(args.test_dataset + "." + args.target_lang + ".translated")

            bleu = translator.get_bleu_for_test_data_translation()
            print("BLEU: {}".format(bleu))

        if args.livetest:
            while True:
                seq = input("Enter sequence: ")
                translator.translate(seq)

        # TODO class for encoder/decoder (model)


# autogenerate docs
# docs 	sphinx-apidoc -o docs nmt
# and probably run Generate docs in pycharm
if __name__ == "__main__":
    main()
