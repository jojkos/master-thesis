# coding: utf-8

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_bleu(reference_file_path, hypothesis_file_path):
    """

    Calculates BLEU score with the reference multi-bleu.perl script from Moses

    Args:
        reference_file_path: path to the reference translation file from the dataset
        hypothesis_file_path: path to the file translated by the translator

    Returns: BLEU score

    """
    logger.info("computing bleu score...")

    with open(hypothesis_file_path, "r", encoding="utf-8") as hypothesis_file:
        args = ["perl", "../scripts/multi-bleu.perl", reference_file_path]

        popen = subprocess.Popen(args, stdin=hypothesis_file)  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE
        popen.wait()
        # output = popen.stdout.read()
        # err_output = popen.stderr.read()
        # print("output:", output)
        # print("error output:", err_output)

        # TODO return the value instead of letting it print it


if __name__ == "__main__":
    get_bleu("../../data/news-commentary-v9.cs-en.en.translated", "../../data/news-commentary-v9.cs-en.en.translated")
