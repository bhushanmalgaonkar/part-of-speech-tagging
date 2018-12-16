from functools import reduce
import numpy as np

from pos_data import read


train = read('data/bc.train')
test = read('data/bc.test')

# Get all unique tags from training dataset and index both ways
unique_tags = set()
for sample in train:
    unique_tags.update(sample[1])

"""
Stores numerical index for each tag
e.g. "noun": 0, "adj": 1, ...
"""
tagIndex = {tag: idx for (idx, tag) in enumerate(unique_tags)}

"""
Opposite of tagIndex
e.g. 0: "noun", 1: "adj", ...
"""
tagValue = {idx: tag for (tag, idx) in tagIndex.items()}


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
emission_probability = [{}] * len(tagIndex)

"""
Transition probability 1 is the probability of having some hidden variable next to some other hidden variable.
E.g. probablity of verb being followed by noun, P(tag_i|tag_i-1)
"""
transition_probability_1 = []

"""
Transition probability 2 is the probability of having some hidden variable given values of 2 previous hidden variables.
E.g. probablity of noun -> verb -> adjective, P(tag_i|tag_i-1, tag_i-2)
"""
transition_probability_2 = {}


def calculate_emission_probability(data):
    emission = {}
    for words, tags in data:
        for idx in range(len(words)):
            if words[idx] in emission_probability[tagIndex]:
                pass
