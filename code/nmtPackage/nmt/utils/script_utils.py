# coding: utf-8

import logging
import subprocess
import os

logger = logging.getLogger(__name__)

SCRIPT_FOLDER = "\\..\\scripts"


def get_script_path(script_name):
    return os.path.dirname(__file__) + SCRIPT_FOLDER + "\\" + script_name


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
        args = ["perl", get_script_path("multi-bleu.perl"), reference_file_path]

        popen = subprocess.Popen(args, stdin=hypothesis_file)  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE
        popen.wait()
        # output = popen.stdout.read()
        # err_output = popen.stderr.read()
        # print("output:", output)
        # print("error output:", err_output)

        # TODO return the value instead of letting it print it


def create_bpe_dataset(paths, symbols):
    args = ["python", get_script_path("subword-nmt\\learn_joint_bpe_and_vocab.py"), "-s", str(symbols),
            "-o", "codes_file"]
    args += ["--input"] + paths
    args += ["--write-vocabulary"]
    args += [path + ".vocab" for path in paths]
    subprocess.run(args)

    for path in paths:
        args = ["python", get_script_path("subword-nmt\\apply_bpe.py"), "-c", "codes_file",
                "--vocabulary", path + ".vocab", "--input", path, "--output", path + ".BPE"]
        subprocess.run(args)
        os.remove(path + ".vocab")

    os.remove("codes_file")


if __name__ == "__main__":
    # get_bleu("data/news-commentary-v9.cs-en.en.translated", "data/news-commentary-v9.cs-en.en.translated")
    create_bpe_dataset([
        get_script_path("subword-nmt\\datasets\\mySmallTest.cs"),
        get_script_path("subword-nmt\\datasets\\mySmallTest.en")
    ], 10)
