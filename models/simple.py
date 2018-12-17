from functools import reduce
import numpy as np
import math
import random

from models.probabilistic import Probabilistic

"""
Simple model

The tag entirely depends on the word itself, so we can choose tag for a given word such that P(tag|word) is maximum.
P(tag|word) = P(word|tag) * P(tag), these two probabilities can be calculated from the data
"""


class Simple(Probabilistic):
    def __init__(self):
        Probabilistic.__init__(self)

    """
    Calculates all the probabilities required for the model to predict new sentences

    Input
    X: list of sentences, where each sentence is list of strings
    y: list of tags associated with X
    """

    def fit(self, X, y):
        self._fetch_tags(y)
        self._calculate_tag_cost(y)
        self._calculate_emission_cost(X, y)

    """
    Calculates best tag for each word of each sentence using P(tag|word) = P(word|tag) * P(tag)

    Input
    X: list of sentences, where each sentence is list of strings

    Output
    list of tags
    """

    def predict(self, X):
        result = []
        for sample in X:
            tags = []
            for word in sample:
                # find tag for word is most likely
                best_tag = None
                min_cost = float('inf')
                for idx in range(len(self.tagIndex)):
                    # P(tag|word) = P(word|tag) * P(tag)
                    tag_given_word = self.get_emission_cost(idx, word) + self.tag_cost[idx]
                    if tag_given_word < min_cost:
                        best_tag = self.tagValue[idx]
                        min_cost = tag_given_word
                tags.append(best_tag)
            result.append(tags)
        return result
