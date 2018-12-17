import math
import numpy as np


class Probabilistic:
    # methods to override

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    # end of methods to override

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
        Negative log of probability that any randomly chosen word will have particular tag, -log(P(tag))
        """
        self.tag_cost = None

        """
        Emission cost is the negative log of probability of having some observed variable given some value of hidden variable.
        In this case, it would be probability of certain word given certain pos tag, P(word_i|tag_i)

        Structure:
        [
            0 (="adj"): {word_i1: count, word_i2: count, ...},
            1 (="adv"): {word_j1: count, word_j2: count, ...},
            ...
        ]
        """
        self.emission_cost = None

        """
        Transition 1 cost is the negative log of probability of having some hidden variable next to some other hidden variable.
        E.g. probablity of verb being followed by noun, P(tag_i|tag_i-1)

        Structure:
        [
            tag_i=0: [
                tag_i-1=0, tag_i-1=1, ...
            ],
            ...
        ]
        """
        self.transition_1_cost = None

        """
        Transition 2 cost is the negative log of probability of having some hidden variable given values of 2 previous hidden variables.
        E.g. probablity of noun -> verb -> adjective, P(tag_i|tag_i-1, tag_i-2)

        Structure:
        [
            tag_i=0: [
                tag_i-1=0: [
                    tag_i-2=0, tag_i-2=1, ...
                ],
                tag_i-1=1, ...
            ],
            ...
        ]
        """
        self.transition_2_cost = None

        """
        Negative log of probability of sentence beginning with particular tag. Array of length same as tags
        """
        self.beginning_cost = None

        """
        Hyperparameters
        """
        self.MISSING_WORD_PROBABILITY = 10e-9
        self.MISSING_WORD_COST = -np.log(self.MISSING_WORD_PROBABILITY)
        self.MISSING_EMISSION_COST = -np.log(10e-9)
        self.MISSING_TRANSITION_1_COST = -np.log(10e-9)
        self.MISSING_TRANSITION_2_COST = -np.log(10e-9)

    # Wrapper functions to handle errors and missing values
    def get_emission_cost(self, tag, word):
        if tag not in self.tagIndex:
            raise Exception(f"Invalid tag: {tag}")

        if word in self.emission_cost[self.tagIndex(tag)]:
            return self.emission_cost[self.tagIndex(tag)][word]
        return self.MISSING_EMISSION_COST

    def get_transition_1_cost(self, tag_i, tag_i_1):
        if tag_i not in self.tagIndex:
            raise Exception(f"Invalid tag: {tag_i}")
        if tag_i_1 not in self.tagIndex:
            raise Exception(f"Invalid tag: {tag_i_1}")

        return self.transition_1_cost[tag_i][tag_i_1]

    def get_transition_2_cost(self, tag_i, tag_i_1, tag_i_2):
        if tag_i not in self.tagIndex:
            raise Exception(f"Invalid tag: {tag_i}")
        if tag_i_1 not in self.tagIndex:
            raise Exception(f"Invalid tag: {tag_i_1}")
        if tag_i_2 not in self.tagIndex:
            raise Exception(f"Invalid tag: {tag_i_2}")

        return self.transition_2_cost[tag_i][tag_i_1][tag_i_2]
    # End of Wrapper functions to handle errors and missing values

    """
    Finds all the unique tags from given list and generates 2 dictionaries: tag->index, index->tag

    Input
    y: list of tags associated with each sentence in training set
    """

    def _fetch_tags(self, y):
        # Get all unique tags from training dataset and index both ways
        unique_tags = set()
        for tags in y:
            unique_tags.update(tags)

        self.tagIndex = {tag: idx for (idx, tag) in enumerate(unique_tags)}
        self.tagValue = {idx: tag for (tag, idx) in self.tagIndex.items()}

    """
    Calculates -log(P(tag)) for each tag

    Input
    y: list of tags associated with each sentence in training set
    """

    def _calculate_tag_cost(self, y):
        count = [0 for _ in range(len(self.tagIndex))]

        # calculate frequency for each word appearing opposite to each tag
        for tags in y:
            for tag in tags:
                count[self.tagIndex[tag]] += 1

        # calculate probability using sum of frequencies of each tag
        # keep all calculations in log
        log_total = math.log(sum(count))
        self.tag_cost = [-(math.log(c) - log_total) for c in count]

    """
    Calculates negative log of emission probability: -log(P(Observed|Hidden))

    In this case, it calculates probablity of some word occuring given some tag, -log(P(word|tag))

    Input
    X: list of sentences, where each sentence is list of strings
    y: list of tags associated with X
    """

    def _calculate_emission_cost(self, X, y):
        self.emission_cost = [{} for _ in range(len(self.tagIndex))]

        # calculate frequency for each word appearing opposite to each tag
        for idx in range(len(X)):
            for w, t in zip(X[idx], y[idx]):
                if w not in self.emission_cost[self.tagIndex[t]]:
                    self.emission_cost[self.tagIndex[t]][w] = 0
                self.emission_cost[self.tagIndex[t]][w] += 1

        # calculate probability using sum of frequencies of words for each tag
        # keep all calculations in log
        for idx in range(len(self.emission_cost)):
            total = sum(self.emission_cost[idx].values())
            self.emission_cost[idx] = {
                k: -(math.log(v) - math.log(total)) for (k, v) in self.emission_cost[idx].items()}

    """
    Calculates negative log of transition probability: -log(P(hidden_t|hidden_t-1))

    In this case, it calculates probablity of some tag given previous tag, -log(P(tag_t|tag_t-1))

    Input
    y: list of tags associated with X
    """

    def _calculate_transition_1_cost(self, y):
        self.transition_1_cost = np.zeros((len(self.tagIndex), len(self.tagIndex)))
        for sample in y:
            for idx in range(1, len(sample)):
                self.transition_1_cost[self.tagIndex[sample[idx]],
                                       self.tagIndex[sample[idx - 1]]] += 1

        # divide by sum to get probabilities
        total = np.sum(self.transition_1_cost, axis=0)
        # to avoid 0/0 errors
        total[total == 0] = np.inf
        self.transition_1_cost /= total

        # give small probability for missing word, so that while testing if this pair appears the probability of tags decreses but does not become 0
        self.transition_1_cost[self.transition_1_cost == 0] = self.MISSING_WORD_PROBABILITY

        # keep calculations in log
        self.transition_1_cost = np.log(self.transition_1_cost)

    """
    Calculates negative log of transition probability: -log(P(hidden_t|hidden_t-1,hidden_t-2))

    In this case, it calculates probablity of some tag given sequence of 2 previous tag, -log(P(tag_t|tag_t-1,tag_t-2))

    Input
    y: list of tags associated with X
    """

    def _calculate_transition_2_cost(self, y):
        self.transition_2_cost = np.zeros(
            (len(self.tagIndex), len(self.tagIndex), len(self.tagIndex)))
        for sample in y:
            for idx in range(2, len(sample)):
                self.transition_2_cost[self.tagIndex[sample[idx]],
                                       self.tagIndex[sample[idx - 1]], self.tagIndex[sample[idx - 2]]] += 1

        # divide by sum to get probabilities
        total = np.sum(self.transition_2_cost, axis=0)
        # to avoid 0/0 errors
        total[total == 0] = np.inf
        self.transition_2_cost /= total

        # give small probability for missing word, so that while testing if this pair appears the probability of tags decreses but does not become 0
        self.transition_2_cost[self.transition_2_cost == 0] = self.MISSING_WORD_PROBABILITY

        # keep calculations in log
        self.transition_2_cost = -np.log(self.transition_2_cost)

    def _calculate_beginning_cost(self, y):
        self.beginning_cost = np.zeros(len(self.tagIndex))
        for sample in y:
            self.beginning_cost[self.tagIndex[sample[0]]] += 1

        # divide by sum to get probabilities
        self.beginning_cost /= np.sum(self.beginning_cost)

        # small probability for missing
        self.beginning_cost[self.beginning_cost == 0] = self.MISSING_WORD_PROBABILITY

        # convert to cost
        self.beginning_cost = -np.log(self.beginning_cost)
