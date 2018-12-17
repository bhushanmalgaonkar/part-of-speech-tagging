#!/usr/bin/env python3

import numpy as np

from models.probabilistic import Probabilistic


"""
The tag depends on word as well as previous tag

Optimizes P(tag_1, tag_2, ... tag_n|word_1, word_2, ... word_n)
    = P(tag_1) * P(tag_2|tag_1) * ... * P(tag_n|tag_n-1) * P(word_1|tag_1) * ... * P(word_n|tag_n)
"""


class HMM(Probabilistic):
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
        self._calculate_transition_1_cost(y)
        self._calculate_beginning_cost(y)

    """
    Calculates maximum a posteriori (MAP) tags for given sentence using Viterbi algorithm

    P(tag_1, tag_2, ... tag_n|word_1, word_2, ... word_n)
        = P(tag_1) * P(tag_2|tag_1) * ... * P(tag_n|tag_n-1) * P(word_1|tag_1) * ... * P(word_n|tag_n)

    Input
    X: list of sentences, where each sentence is list of strings

    Output
    list of tags with minimum cost, i.e. maximum posterior probability
    """

    def predict(self, X):
        result = []
        for sample in X:
            cost = np.full((len(sample), len(self.tagIndex)), np.inf)
            cost[0, :] = self.beginning_cost
            cost[0, :] += [self.get_emission_cost(idx, sample[0])
                           for idx in range(len(self.tagIndex))]

            # fill the dp table
            for wi in range(1, len(sample)):
                for ti in range(len(self.tagIndex)):
                    for pti in range(len(self.tagIndex)):
                        cost[wi, ti] = min(
                            cost[wi, ti], cost[wi - 1, pti] + self.get_transition_1_cost(ti, pti) + self.get_emission_cost(ti, sample[wi]))

            # backtrack to get tags that result in minimum cost
            tags = [-1 for _ in range(len(sample))]
            tags[-1] = np.argmin(cost[-1, :])
            for wi in range(len(sample) - 1, 0, -1):
                # find which previous tag lies on shortest path
                for pti in range(len(self.tagIndex)):
                    if cost[wi - 1, pti] + self.get_transition_1_cost(tags[wi], pti) + self.get_emission_cost(tags[wi], sample[wi]) == cost[wi, tags[wi]]:
                        tags[wi - 1] = pti
                        break

            # return list of tags associated with indexes
            result.append([self.tagValue[t] for t in tags])
        return result
