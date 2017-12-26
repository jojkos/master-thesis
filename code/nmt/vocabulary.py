from nltk import FreqDist
from nmt import SpecialSymbols
import numpy as np


class Vocabulary(object):
    def __init__(self, word_seq, max_vocab_size, add_specials=True):
        # Creating the vocabulary set with the most common words

        # cannot use keras tokenizer, because we need to add our SpecialSymbols in the vocabuly and keras don't do that
        dist = FreqDist(np.hstack(word_seq))
        vocab = dist.most_common(max_vocab_size)

        # Creating an array of words from the vocabulary set,
        # we will use this array as index-to-word dictionary
        self.ix_to_word = [word[0] for word in vocab]
        # ADD special vocabulary symbols at the start
        if add_specials:
            self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.PAD, SpecialSymbols.PAD_IX)
            self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.GO, SpecialSymbols.GO_IX)
            self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.EOS, SpecialSymbols.EOS_IX)
            self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.UNK, SpecialSymbols.UNK_IX)

        self.ix_to_word = {index: word for index, word in enumerate(self.ix_to_word)}
        self.word_to_ix = {self.ix_to_word[ix]: ix for ix in self.ix_to_word}
        # TODO how to use pretrained embedding with these custom symbols
        # https://github.com/fchollet/keras/issues/6480
        # https://github.com/fchollet/keras/issues/3325
        self.vocab_len = len(self.ix_to_word)

    @staticmethod
    def _insert_symbol_to_vocab(vocab, symbol, index):
        """
        symbol can potentially (for instance as a result of tokenizing where _go and _eos are added to sequence)
        be already part of vocabulary, but we want it to be on specific index
        """

        if symbol in vocab:
            vocab.remove(symbol)

        vocab.insert(index, symbol)

        return vocab

    def get_word(self, ix):
        return self.ix_to_word[ix]

    def get_index(self, word):
        return self.word_to_ix[word]
