from functools import reduce
import numpy as np
import math
import random


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

    def fit(self, X, y):
        # Get all unique tags from training dataset and index both ways
        unique_tags = set()
        for tags in y:
            unique_tags.update(tags)

        self.tagIndex = {tag: idx for (idx, tag) in enumerate(unique_tags)}
        self.tagValue = {idx: tag for (tag, idx) in self.tagIndex.items()}

        self._calculate_emission_probability(X, y)
        # self._calculate_transition_1_probability(X, y)

    def predict(self, X):
        result = []
        for sample in X:
            tags = []
            for word in sample:
                # find tag for word is most likely
                best_tag = None
                max_prob = -float('inf')
                for idx in range(len(self.tagIndex)):
                    if word in self.emission_probability[idx] and self.emission_probability[idx][word] > max_prob:
                        best_tag = self.tagValue[idx]
                        max_prob = self.emission_probability[idx][word]
                # if we couldn't find this word in any of the tags, just return random tag
                if best_tag is None:
                    best_tag = self.tagValue[random.randint(0, len(self.tagValue)-1)]
                tags.append(best_tag)
            result.append(tags)
        return result

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

    def _calculate_transition_1_probability(self, training_data):
        pass

    def _calculate_transition_2_probability(self, training_data):
        pass
