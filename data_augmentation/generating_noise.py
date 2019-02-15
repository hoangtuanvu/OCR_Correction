from random import randint, choice
from data_augmentation.constants import SOURCE_PATH, TARGET_PATH

import numpy as np


def add_char_randomly(input_str, vocabulary):
    """

    :param input_str: input text
    :param vocabulary: contains all characters
    :return: new text from input_str after inserting new character randomly
    """
    random_char = choice(list(vocabulary))
    i = randint(a=0, b=len(input_str) - 1)
    output_str = input_str[0:i] + random_char + input_str[i:]
    assert len(output_str) == len(input_str) + 1
    assert random_char in output_str
    return output_str


def remove_char_randomly(input_str):
    """

    :param input_str: input text
    :return: new text from input_str after removing a character randomly
    """
    i = randint(a=0, b=len(input_str) - 1)
    char_to_remove = input_str[i]
    before = input_str.count(char_to_remove)
    output_str = input_str[0:i] + input_str[i + 1:]
    after = output_str.count(char_to_remove)
    assert before - 1 == after
    assert len(output_str) == len(input_str) - 1
    return output_str


def change_char_randomly(input_str, vocabulary):
    """

    :param input_str: input text
    :param vocabulary: contains all characters
    :return: new text from input_str after changing a character from input by another one from vocabulary randomly
    """
    random_char = choice(list(vocabulary))
    i = randint(a=0, b=len(input_str) - 1)
    output_str = input_str[0:i] + random_char + input_str[i + 1:]
    assert len(output_str) == len(input_str)
    assert random_char in output_str
    return output_str


def permute_two_chars_randomly(input_str):
    """

    :param input_str: input text
    :return: new text from input_str after swapping 2 characters randomly
    """

    def _swap(s, i, j):
        c = list(s)
        c[i], c[j] = c[j], c[i]
        return ''.join(c)

    # interest is limited here.
    i1 = randint(a=0, b=len(input_str) - 1)
    i2 = randint(a=0, b=len(input_str) - 1)
    output_str = _swap(input_str, i1, i2)
    return output_str


def add_noise_to_data(input_str, probs, vocabulary):
    no_change, add_char, remove_char, change_char = 0, 1, 2, 3
    noise_type = np.random.choice(a=[no_change, add_char, remove_char, change_char], size=1, replace=False, p=probs)
    if noise_type == no_change:
        return input_str, no_change
    elif noise_type == add_char:
        return add_char_randomly(input_str, vocabulary), add_char
    elif noise_type == remove_char:
        return remove_char_randomly(input_str), remove_char
    elif noise_type == change_char:
        return change_char_randomly(input_str, vocabulary), change_char
    else:
        raise Exception('Invalid return.')


def build_vocabulary():
    vocabulary = set()

    def load_database(data_path):
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                for element in line:
                    vocabulary.add(element)

    load_database(SOURCE_PATH)
    load_database(TARGET_PATH)

    vocabulary = sorted(list(vocabulary))
    print("Vocab size: {}".format(len(vocabulary)))

    return vocabulary


if __name__ == "__main__":
    print(build_vocabulary())
