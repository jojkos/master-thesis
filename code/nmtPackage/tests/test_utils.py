from nmt import utils
import os
import shutil


def test_read_file_to_lines():
    file_path = "testfile.txt"
    text = "12345\n67890\nabcde"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    lines = utils.read_file_to_lines(file_path, 2)

    os.remove(file_path)

    assert lines == ["12345", "67890"]


def test_get_bucket_ix():
    ix = utils.get_bucket_ix(5, 2)
    assert ix == 3

    ix = utils.get_bucket_ix(2, 2)
    assert ix == 1

    ix = utils.get_bucket_ix(10, 3)
    assert ix == 4


def test_split_to_buckets():
    # 1st test
    x_word_seq = [
        ["1"],

    ]
    y_word_seq = [
        ["1", "2"]
    ]
    test_buckets = {
        1: {"x_word_seq": [["1"]], "y_word_seq": [["1", "2"]], "x_max_seq_len": 1, "y_max_seq_len": 2}
    }
    buckets = utils.split_to_buckets(x_word_seq, y_word_seq, bucket_range=2, bucket_min_size=2)

    assert buckets == test_buckets

    # 2nd test
    x_word_seq = [
        ["1"],
        ["1", "2"],
        ["1", "2", "3"],
        ["1", "2", "3", "4"],
        ["1", "2", "3", "4", "5"],
        ["1", "2", "3", "4", "5"],
        ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ]

    y_word_seq = [
        ["1"],
        ["1", "2"],
        ["1", "2"],
        ["1", "2", "3", "4"],
        ["1", "2", "3", "4", "5"],
        ["1", "2", "3", "4", "5", "6", "7"],
        ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ]

    test_buckets = {
        1: {"x_word_seq": [["1"], ["1", "2"]], "y_word_seq": [["1"], ["1", "2"]], "x_max_seq_len": 2,
            "y_max_seq_len": 2},
        2: {"x_word_seq": [["1", "2", "3"], ["1", "2", "3", "4"]], "y_word_seq": [["1", "2"], ["1", "2", "3", "4"]],
            "x_max_seq_len": 4, "y_max_seq_len": 4},
        4: {"x_word_seq": [["1", "2", "3", "4", "5"], ["1", "2", "3", "4", "5"],
                           ["1", "2", "3", "4", "5", "6", "7", "8", "9"]],
            "y_word_seq": [["1", "2", "3", "4", "5", "6", "7"], ["1", "2", "3", "4", "5"],
                           ["1", "2", "3", "4", "5", "6", "7", "8", "9"]],
            "x_max_seq_len": 9, "y_max_seq_len": 9}
    }

    buckets = utils.split_to_buckets(x_word_seq, y_word_seq, bucket_range=2, bucket_min_size=2)

    assert buckets == test_buckets


def test_prepare_folder():
    path = "./testFolder"

    # remove, if for some reasing, folder already exists
    if os.path.exists(path):
        shutil.rmtree(path)
        os.rmdir(path)

    utils.prepare_folder(path, False)

    assert os.path.isdir(path)

    # clean
    os.rmdir(path)


def test_prepare_folders():
    paths = ["./testFolder", "./testFolder2"]

    # remove, if for some reasing, folder already exists
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.rmdir(path)

    utils.prepare_folders(paths, False)

    for path in paths:
        assert os.path.isdir(path)
        # clean
        os.rmdir(path)


def test_tokenize():
    lines = ["first sentence!", "second sentence."]
    tokenized = utils.tokenize(lines)
    assert tokenized == [["first", "sentence"], ["second", "sentence"]]


def test_split_lines():
    lines = ["first sentence", "second sentence"]
    tokenized = utils.split_lines(lines)
    assert tokenized == [["first", "sentence"], ["second", "sentence"]]


def test_incremental_average():
    avg = 0
    avg = utils.incremental_average(avg, 1, 1)
    assert avg == 1

    avg = utils.incremental_average(avg, 1, 2)
    assert avg == 1

    avg = utils.incremental_average(avg, 2, 3)
    assert round(avg, 2) == 1.33

    avg = utils.incremental_average(avg, 4, 4)
    assert avg == 2
