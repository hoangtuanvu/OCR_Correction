from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nlc_data
import random
import numpy as np
import tensorflow as tf
from nltk.metrics.distance import edit_distance
import codecs
from charguana import get_charset

FLAGS = tf.app.flags.FLAGS


def pair_iter(fnamex, fnamey, batch_size, num_layers, sort_and_shuffle=True):
    fdx, fdy = open(fnamex), open(fnamey)
    batches = []

    while True:
        if len(batches) == 0:
            fill_batch(batches, fdx, fdy, batch_size, sort_and_shuffle=sort_and_shuffle)
        if len(batches) == 0:
            break

        x_tokens, y_tokens = batches.pop(0)
        y_tokens = add_sos_eos(y_tokens)

        x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)

        source_tokens = np.array(x_padded).T
        source_mask = (source_tokens != nlc_data.PAD_ID).astype(np.int32)

        target_tokens = np.array(y_padded).T
        target_mask = (target_tokens != nlc_data.PAD_ID).astype(np.int32)

        yield (source_tokens, source_mask, target_tokens, target_mask)


def fill_batch(batches, fdx, fdy, batch_size, sort_and_shuffle=True):
    def tokenize(string):
        return [int(s) for s in string.split()]

    line_pairs = []
    linex, liney = fdx.readline(), fdy.readline()

    while linex and liney:
        x_tokens, y_tokens = tokenize(linex), tokenize(liney)

        if len(x_tokens) < FLAGS.max_seq_len and len(y_tokens) < FLAGS.max_seq_len:
            line_pairs.append((x_tokens, y_tokens))
        if len(line_pairs) == batch_size * 16:
            break
        linex, liney = fdx.readline(), fdy.readline()

    if sort_and_shuffle:
        line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

    for batch_start in range(0, len(line_pairs), batch_size):
        x_batch, y_batch = list(zip(*line_pairs[batch_start:batch_start + batch_size]))
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)


def add_sos_eos(tokens):
    return list(map(lambda token_list: [nlc_data.SOS_ID] + token_list + [nlc_data.EOS_ID], tokens))


def padded(tokens, depth):
    maxlen = max(list(map(lambda x: len(x), tokens)))
    align = pow(2, depth - 1)
    padlen = maxlen + (align - maxlen) % align
    return list(map(lambda token_list: token_list + [nlc_data.PAD_ID] * (padlen - len(token_list)), tokens))


def evaluate(sentx, senty):
    sent_max_len = max(len(list(sentx)), len(list(senty)))
    if sent_max_len == 0:
        return 0

    return edit_distance(sentx, senty) / sent_max_len


def write_anomaly(input_tokens, vocab_path, output_path):
    rev_vocab = []
    with codecs.open(vocab_path, mode="r", encoding="utf-8") as fr:
        rev_vocab.extend(fr.readlines())
    rev_vocab = [line.strip("\n") for line in rev_vocab]
    vocab = dict([(y, x) for (y, x) in enumerate(rev_vocab)])

    a_new = []
    for item in input_tokens.T:
        tmp = ''.join([vocab[x] for x in item])
        a_new.append(tmp)

    with open(output_path, 'w') as f:
        for item in a_new:
            f.write(item + '\n')


def remove_same_pairs(source_path, target_path, source_out, target_out):
    tmp = {}
    f_source_out = open(source_out, 'w')
    f_target_out = open(target_out, 'w')

    with open(source_path, 'r') as f_source:
        with open(target_path, 'r') as f_target:
            for (sent1, sent2) in zip(f_source, f_target):
                sent1 = sent1.strip()
                sent2 = sent2.strip()

                if (sent1 + '|' + sent2) not in tmp:
                    tmp[sent1 + '|' + sent2] = 1
                    f_source_out.write(sent1 + '\n')
                    f_target_out.write(sent2 + '\n')


def get_dict(source_path, target_path):
    a = {}
    with open(source_path, 'r') as f1:
        with open(target_path, 'r') as f2:
            for (sent1, sent2) in zip(f1, f2):
                sent1 = sent1.strip()
                sent2 = sent2.strip()
                a[sent1 + '|' + sent2] = 1
    return a


def get_hira_vocab():
    tmp = []
    for ch in list(set(get_charset('hiragana'))):
        tmp.append(ch)
    tmp = list(set(tmp))
    return tmp
