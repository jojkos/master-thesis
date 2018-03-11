import logging
from nmt import SpecialSymbols
import numpy as np

logger = logging.getLogger(__name__)


class Candidate(object):
    """
        Temporary candidate or hypotheses of the translated sequence used in beam search
    """

    def __init__(self, last_prediction, decoded_sentence, states_value, score, sampled_word=None):
        # Generate empty target sequence of length 1.
        self.last_prediction = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        self.last_prediction[0, 0] = last_prediction

        self.decoded_sentence = decoded_sentence
        self.score = score  # the lower, the better
        self.states_value = states_value
        self.decoded_sentence = decoded_sentence
        self.finalised = False

        if sampled_word == SpecialSymbols.EOS:
            self.finalise()
        elif sampled_word:
            self.decoded_sentence += sampled_word + " "

    def finalise(self):
        self.decoded_sentence = self.decoded_sentence.strip()
        self.finalised = True

    def get_sentence_length(self):
        return len(self.decoded_sentence.split(" "))