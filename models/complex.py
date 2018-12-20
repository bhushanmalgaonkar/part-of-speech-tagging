#!/usr/bin/env python3

import numpy as np

from models.probabilistic import Probabilistic


"""
The tag depends on word as well as previous tag, and tag previous to that

Here we optimize P(tag_1, tag_2, ... tag_n|word_1, word_2, ... word_n)
    = P(tag_1) * P(tag_2|tag_1) * ... * P(tag_n|tag_n-1) * P(word_1|tag_1) * ... * P(word_n|tag_n)

Optimization is done using Marcov Chain Monte Carlo with Gibbs sampling.
"""


class Complex(Probabilistic):
    def __init__(self):
        Probabilistic.__init__(self)

        # Hyperparameters
        self.NUM_GIBBS_ITER = 150

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
        self._calculate_transition_1_cost(y)
        self._calculate_transition_2_cost(y)

    def predict(self, X):
        result = []
        for sample in X:
            # randomly initilize tags
            tags = np.random.choice(np.arange(len(self.tagIndex)), len(sample))

            for iter in range(self.NUM_GIBBS_ITER):
                # Choose a word at random for which we will find best tag
                word_idx = np.random.randint(0, len(sample))

                prob_dist = self._calculate_probability_dist(tags, sample, word_idx)

                # Select randomly over given new probability distribution
                tag_idx = np.random.choice(np.arange(len(prob_dist)), p=prob_dist)
                tags[word_idx] = tag_idx

            # sampling may randomly (with very low probability) pick very unlikely tag for a particular word, therefore
            # return most likely tag for each word using generated posterior probabilities
            for iter in range(5):
                for idx in range(len(tags)):
                    tags[idx] = np.argmax(self._calculate_probability_dist(tags, sample, idx))

            result.append([self.tagValue[t] for t in tags])
        return result

    """
    Calculates probability distribution over each tag assigned to word at index i, keeping all other tags same.
    Returns an array of "probabilities" for each tag as per indexes specified in self.tagIndex
    """

    def _calculate_probability_dist(self, tags, sentence, word_idx):
        distribution = np.zeros(len(self.tagIndex))
        original_tag = tags[word_idx]

        for tag_idx in range(len(self.tagIndex)):
            tags[word_idx] = tag_idx

            # store the value at index corresponding to that tag
            distribution[tag_idx] = self._calculate_posterior(tags, sentence)

        tags[word_idx] = original_tag

        # Convert likelyhood into probability distribution to make sampling easier
        # subtracting minimum is same as dividing probabilities, which keeps the relative probabilities the same
        # 10 times more likely event will still be 10 times more likely
        distribution -= np.min(distribution)

        # convert logs into probabilities, since now they are within reasonable bounds
        distribution = np.exp(-distribution)
        distribution /= sum(distribution)

        return distribution

    """
    Calculates likelyhood of tags (S1, S2, S3, S4...) for sentence (W1, W2, W3, W4...) using bayes theorem
    likelyhood = P(S1) * P(S2|S1) * P(S3|S2,S1) * P(S4|S3,S2) * ...
                    * P(S1|W1) * P(S2|W2) * P(S3|W3) * P(S4|W4) * ...
    To keep calculations within bounds use -log of all probabilities
    """

    def _calculate_posterior(self, tags, sentence):
        posterior = 0

        if len(sentence) > 0:
            posterior += self.tag_cost[tags[0]]
        if len(sentence) > 1:
            posterior += self.get_transition_1_cost(tags[1], tags[0])
        for i in range(2, len(sentence)):
            posterior += self.get_transition_2_cost(tags[i], tags[i - 1], tags[i - 2])

        for i in range(len(sentence)):
            posterior += self.get_emission_cost(tags[i], sentence[i])

        return posterior
