import os


"""
Reads file of the following form
sentence1_word1 sentence1_tag1 sentence1_word2 sentence1_tag2 ...
sentence2_word1 sentence2_tag1 sentence2_word2 sentence2_tag2 ...

Returns data in the following form
list[
    tuple(      # first sentence
        list[   # words in first sentence in order
            sentence1_word1, sentence1_word2, ...
        ],
        list[   # tags of first sentence in order
            sentence1_tag1, sentence1_tag2, ...
        ]
    ),
    ...
]
"""


def read(filepath):
    X, y = [], []
    if not os.path.exists(filepath):
        return data

    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            split = line.lower().split()
            if len(split) % 2 != 0:
                raise Exception(
                    f"Invalid sample on line {i}: Number of words and tags do not match")
            X.append(split[::2])
            y.append(split[1::2])
    return X, y
