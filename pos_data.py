import os


"""
mode = 'train':
Reads file of the following form
sentence1_word1 sentence1_tag1 sentence1_word2 sentence1_tag2 ...
sentence2_word1 sentence2_tag1 sentence2_word2 sentence2_tag2 ...

Returns data in the following form: 2 lists, 1st for words, 2nd for tags
X, y =
list[
    list[   # words in first sentence in order
        word1, word2, ...
    ],
    list[   # words of second sentence in order
        word1, word2, ...
    ],
    ...
],
list[
    list[   # tags in first sentence in order
        tag1, tag2, ...
    ],
    list[   # tags of second sentence in order
        tag1, tag2, ...
    ],
    ...
]


mode = 'test':
Reads file of the following form
sentence1_word1 sentence1_word2 ...
sentence2_word1 sentence2_word2 ...

Returns single list of sentences
X =
list[
    list[   # words in first sentence in order
        word1, word2, ...
    ],
    list[   # words of second sentence in order
        word1, word2, ...
    ],
    ...
]
"""


def read(filepath, mode):
    if mode not in ['train', 'test']:
        raise Exception(f'Invalid mode {mode}. Use \'train\' or \'test\'')

    X, y = [], []
    if not os.path.exists(filepath):
        return data

    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            split = line.lower().split()
            if mode == 'train':
                if len(split) % 2 != 0:
                    raise Exception(
                        f"Invalid sample on line {i}: Number of words and tags do not match")
                X.append(split[::2])
                y.append(split[1::2])
            elif mode == 'test':
                X.append(split)

    if mode == 'train':
        return (X, y)
    elif mode == 'test':
        return (X,)
    return None
