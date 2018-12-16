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
        Transition probability 1 is the probability of having some hidden variable next to some other hidden variable.
        E.g. probablity of verb being followed by noun, P(tag_i|tag_i-1)
        """
        self.transition_1_probability = None

        """
        Transition probability 2 is the probability of having some hidden variable given values of 2 previous hidden variables.
        E.g. probablity of noun -> verb -> adjective, P(tag_i|tag_i-1, tag_i-2)
        """
        self.transition_2_probability = None

        """
        Constants
        """
        self.MISSING_WORD_PROBABILITY = math.log(10e-9)

    def fit(self, X, y):
        # Get all unique tags from training dataset and index both ways
        unique_tags = set()
        for tags in y:
            unique_tags.update(tags)

        self.tagIndex = {tag: idx for (idx, tag) in enumerate(unique_tags)}
        self.tagValue = {idx: tag for (tag, idx) in self.tagIndex.items()}

        self._calculate_tag_probability(y)
        self._calculate_emission_probability(X, y)

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
                    prob = word_given_tag + self.tag_probability[idx]
                    if prob > max_prob:
                        best_tag = self.tagValue[idx]
                        max_prob = prob

                # if we couldn't find this word in any of the tags, just return random tag
                if best_tag is None:
                    best_tag = self.tagValue[random.randint(0, len(self.tagValue)-1)]

                tags.append(best_tag)
            result.append(tags)
        return result

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
