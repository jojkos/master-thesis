import logging

import nmt.utils as utils

logger = logging.getLogger(__name__)


class Dataset(object):
    """
        Class encapsuling loading of the dataset from file
    """
    def __init__(self, dataset_path, source_lang, target_lang, num_samples, tokenize):
        """

        Args:
            dataset_path (str):
            source_lang (str):
            target_lang (str):
            num_samples (str):
            tokenize (bool):
        """
        self.dataset_path = dataset_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.num_samples = num_samples

        self._prepare_dataset(tokenize)

        self.num_samples = len(self.x_word_seq)

        assert len(self.x_word_seq) == len(self.y_word_seq), \
            "dataset %s - has different number of source and target sequences %s %s" % (
            dataset_path, len(self.x_word_seq), len(self.y_word_seq))

    def _prepare_dataset(self, tokenize):
        """

        Loads both dataset files and stores them as sequences. Stores max seq lens as well.

        Args:
            tokenize (bool):

        """
        x_file_path = "{}.{}".format(self.dataset_path, self.source_lang)
        x_lines = utils.read_file_to_lines(x_file_path, self.num_samples)

        y_file_path = "{}.{}".format(self.dataset_path, self.target_lang)
        y_lines = utils.read_file_to_lines(y_file_path, self.num_samples)

        # it seems that some corpuses (like WMT news commentary) has some lines empty for source or target language

        if tokenize:
            self.x_word_seq, self.y_word_seq, self.x_max_seq_len, self.y_max_seq_len = utils.tokenize(x_lines, y_lines)
        else:
            self.x_word_seq, self.x_max_seq_len = utils.split_lines(x_lines)
            self.y_word_seq, self.y_max_seq_len = utils.split_lines(y_lines)


# in folder code
# python -m nmt.dataset
if __name__ == "__main__":
    dataset = Dataset("data/news-commentary-v9.cs-en",
                      "cs", "en", 100, True)
    print(dataset.x_max_seq_len)
    print(len(dataset.x_word_seq))
