from functools import reduce
import numpy as np
import math
import random

"""
Simple model

The tag entirely depends on the word itself, so we can choose tag for a given word such that P(tag|word) is maximum.
P(tag|word) = P(word|tag) * P(tag), these two probabilities can be calculated from the data
"""


class Simple:
    def __init__(self):
        """
        Stores numerical index for each tag
        e.g. "noun": 0, "adj": 1, ...
        """
        self.tagIndex = None

        """
        Opposite of tagIndex
        e.g. 0: "noun", 1: "adj", ...
        """
        self.tagValue = None

        """
        Probability that any randomly chosen word will have particular tag, P(tag)
        """
        self.tag_probability = None

        """
        Emission probability is the probability of having some observed variable given some value of hidden variable.
        In this case, it would be probability of certain word given certain pos tag, P(word_i|tag_i)

        Structure:
        [
            0 (="adj"): {word_i1: count, word_i2: count, ...},
            1 (="adv"): {word_j1: count, word_j2: count, ...},
            ...
        ]
        """
        self.emission_probability = None

        """
        Constants
        """
        self.MISSING_WORD_PROBABILITY = math.log(10e-9)

    """
    Calculates all the probabilities required for the model to predict new sentences

    Input
    X: list of sentences, where each sentence is list of strings
    y: list of tags associated with X
    """

    def fit(self, X, y):
        # Get all unique tags from training dataset and index both ways
        unique_tags = set()
        for tags in y:
            unique_tags.update(tags)

        self.tagIndex = {tag: idx for (idx, tag) in enumerate(unique_tags)}
        self.tagValue = {idx: tag for (tag, idx) in self.tagIndex.items()}

        self._calculate_tag_probability(y)
        self._calculate_emission_probability(X, y)

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
                max_prob = -float('inf')
                for idx in range(len(self.tagIndex)):
                    word_given_tag = self.emission_probability[idx][
                        word] if word in self.emission_probability[idx] else self.MISSING_WORD_PROBABILITY

                    # P(tag|word) = P(word|tag) * P(tag)
                    tag_given_word = word_given_tag + self.tag_probability[idx]
                    if tag_given_word > max_prob:
                        best_tag = self.tagValue[idx]
                        max_prob = tag_given_word
                tags.append(best_tag)
            result.append(tags)
        return result

    """
    Calculates P(tag) for each tag

    Input
    y: list of tags associated with each sentence in training set
    """

    def _calculate_tag_probability(self, y):
        count = [0 for _ in range(len(self.tagIndex))]

        # calculate frequency for each word appearing opposite to each tag
        for tags in y:
            for tag in tags:
                count[self.tagIndex[tag]] += 1

        # calculate probability using sum of frequencies of each tag
        # keep all calculations in log
        log_total = math.log(sum(count))
        self.tag_probability = [(math.log(c) - log_total) for c in count]

    """
    Calculates emission probability: P(Observed|Hidden)

    In this case, it calculates probablity of some word occuring given some tag, P(word|tag)

    Input
    X: list of sentences, where each sentence is list of strings
    y: list of tags associated with X
    """

    def _calculate_emission_probability(self, X, y):
        self.emission_probability = [{} for _ in range(len(self.tagIndex))]

        # calculate frequency for each word appearing opposite to each tag
        for idx in range(len(X)):
            for w, t in zip(X[idx], y[idx]):
                if w not in self.emission_probability[self.tagIndex[t]]:
                    self.emission_probability[self.tagIndex[t]][w] = 0
                self.emission_probability[self.tagIndex[t]][w] += 1

        # calculate probability using sum of frequencies of words for each tag
        # keep all calculations in log
        for idx in range(len(self.emission_probability)):
            total = sum(self.emission_probability[idx].values())
            self.emission_probability[idx] = {
                k: (math.log(v) - math.log(total)) for (k, v) in self.emission_probability[idx].items()}
