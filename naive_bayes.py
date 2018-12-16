from functools import reduce
import numpy as np

from pos_data import read


class NaiveBayes:
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

    def fit(self, training_data):
        # Get all unique tags from training dataset and index both ways
        unique_tags = set()
        for _, tags in training_data:
            unique_tags.update(tags)

        self.tagIndex = {tag: idx for (idx, tag) in enumerate(unique_tags)}
        self.tagValue = {idx: tag for (tag, idx) in self.tagIndex.items()}

        self._calculate_emission_probability(training_data)
        self._calculate_transition_1_probability(training_data)

    def predict(self, testing_data):
        pass

    def _calculate_emission_probability(self, training_data):
        self.emission_probability = [{} for _ in range(len(self.tagIndex))]
        # calculate frequency for each word appearing opposite to each tag
        for words, tags in training_data:
            for w, t in zip(words, tags):
                if w not in self.emission_probability[self.tagIndex[t]]:
                    self.emission_probability[self.tagIndex[t]][w] = 0
                self.emission_probability[self.tagIndex[t]][w] += 1

        # calculate probability using sum of frequencies of words for each tag
        for idx in range(len(self.emission_probability)):
            total = sum(self.emission_probability[idx].values())
            self.emission_probability[idx] = {
                k: v/total for (k, v) in self.emission_probability[idx].items()}

    def _calculate_transition_1_probability(self, training_data):
        pass

    def _calculate_transition_2_probability(self, training_data):
        pass


train = read('data/bc.train')
test = read('data/bc.test')

nb = NaiveBayes()
nb.fit(train)
