# coding: utf-8

import logging
import subprocess
import os

logger = logging.getLogger(__name__)

SCRIPT_FOLDER = "/../scripts"


def get_script_path(script_name):
    return os.path.dirname(__file__) + SCRIPT_FOLDER + "/" + script_name


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
    """
    Learns and applies BPE (https://github.com/rsennrich/subword-nmt) on merged vocabulary from each file in paths
    Args:
        paths: array of paths to files with lines of sentences
        symbols: how many symbols to learn

    Example:
         script_utils.create_bpe_dataset(["G:/Clouds/DPbigFiles/WMT17/devSet/newstest2015-csen.cs", "G:/Clouds/DPbigFiles/WMT17/devSet/newstest2015-csen.en"], 10)
    """
    codes_path = os.path.dirname(paths[0]) + "/codesfile"

    args = ["python", get_script_path("subword-nmt/learn_joint_bpe_and_vocab.py"), "-s", str(symbols),
            "-o", codes_path]
    args += ["--input"] + paths
    args += ["--write-vocabulary"]
    args += [path + ".vocab" for path in paths]
    subprocess.run(args)

    for path in paths:
        args = ["python", get_script_path("subword-nmt/apply_bpe.py"), "-c", codes_path,
                "--vocabulary", path + ".vocab", "--input", path, "--output", path + ".BPE"]
        subprocess.run(args)


def create_bpe_testdataset(paths, vocab_paths, codefile_path):
    """
    Applies BPE to test dataset, based on vocabs and codes learnt from the original dataset
    Args:
        paths: array of paths to test dataset files with lines of sentences
        vocab_paths: paths to vocab files created by create_bpe_dataset in corresponding order to paths
        codefile_path: path to codes file created by create_bpe_dataset

    Example:
        script_utils.create_bpe_testdataset(["G:/Clouds/DPbigFiles/WMT17/commonCrawl/commoncrawl.cs-en.cs", "G:/Clouds/DPbigFiles/WMT17/commonCrawl/commoncrawl.cs-en.en"], ["G:/Clouds/DPbigFiles/WMT17/devSet/newstest2015-csen.cs.vocab", "G:/Clouds/DPbigFiles/WMT17/devSet/newstest2015-csen.en.vocab"], "G:/Clouds/DPbigFiles/WMT17/devSet/codesfile")
    """

    assert len(paths) == len(vocab_paths)

    for ix, path in enumerate(paths):
        args = ["python", get_script_path("subword-nmt/apply_bpe.py"), "-c", codefile_path,
                "--vocabulary", vocab_paths[ix], "--input", path, "--output", path + ".BPE"]
        subprocess.run(args)


if __name__ == "__main__":
    # get_bleu("data/news-commentary-v9.cs-en.en.translated", "data/news-commentary-v9.cs-en.en.translated")
    create_bpe_dataset([
        get_script_path("subword-nmt/datasets/mySmallTest.cs"),
        get_script_path("subword-nmt/datasets/mySmallTest.en")
    ], 10)
