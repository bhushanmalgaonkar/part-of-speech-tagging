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
    data = []
    if not os.path.exists(filepath):
        return data

    with open(filepath) as f:
        for line in f.readlines():
            split = line.lower().split()
            data.append((split[::2], split[1::2]))
    return data
