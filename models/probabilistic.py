import math


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

        Structure:

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

    """
    Calculates transition probability: P(hidden_t|hidden_t-1)

    In this case, it calculates probablity of some tag given previous tag, P(tag_t|tag_t-1)

    Input
    y: list of tags associated with X
    """

    def _calculate_transition_1_probability(self, y):
        pass

    """
    Calculates transition probability: P(hidden_t|hidden_t-1,hidden_t-2)

    In this case, it calculates probablity of some tag given sequence of 2 previous tag, P(tag_t|tag_t-1,tag_t-2)

    Input
    y: list of tags associated with X
    """

    def _calculate_transition_2_probability(self, y):
        pass
