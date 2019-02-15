from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kenlm
import numpy as np
import tensorflow as tf

import nlc_data
import nlc_model
import argparse

tf.app.flags.DEFINE_float("learning_rate", 0.00015, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 5000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 50, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "./data_dir/data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./data_dir/train_data", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_string("lmfile", "./data_dir/kenlm/corpus.binary", "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0.1, "Language model relative weight.")

FLAGS = tf.app.flags.FLAGS
reverse_vocab, vocab, vocab_size = None, None, None
lm = None
model, sess = None, None


def create_model(session, vocab_size, forward_only):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def tokenize(sent, vocab, depth=FLAGS.num_layers):
    align = pow(2, depth - 1)
    token_ids = nlc_data.sentence_to_token_ids(sent, vocab, nlc_data.char_tokenizer)
    ones = [1] * len(token_ids)
    pad = (align - len(token_ids)) % align

    token_ids += [nlc_data.PAD_ID] * pad
    ones += [0] * pad

    source = np.array(token_ids).reshape([-1, 1])
    mask = np.array(ones).reshape([-1, 1])

    return source, mask


def de_tokenize(sents, reverse_vocab):
    # TODO: char vs word
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(nlc_data._START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent

    return [detok_sent(s) for s in sents]


def lm_rank(strs, probs, top_n):
    if lm is None:
        return strs[0]
    a = FLAGS.alpha
    lmscores = [lm.score(s) / (1 + len(s)) for s in strs]
    probs = [p / (len(s) + 1) for (s, p) in zip(strs, probs)]
    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]

    generated = [[strs[item], probs[item]] for item in rerank[len(rerank) - top_n:]]
    generated.reverse()

    return generated


def decode_beam(model, sess, encoder_output, max_beam_size):
    toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
    return toks.tolist(), probs.tolist()


def fix_sent(model, sess, sent, top_n):
    # Tokenize
    input_toks, mask = tokenize(sent, vocab)

    # Encode
    encoder_output = model.encode(sess, input_toks, mask)

    # Decode
    beam_toks, probs = decode_beam(model, sess, encoder_output, FLAGS.beam_size)

    # De-tokenize
    beam_strs = de_tokenize(beam_toks, reverse_vocab)

    # Language Model ranking
    best_str = lm_rank(beam_strs, probs, top_n)
    # Return
    return best_str


def load_vocab():
    # Prepare NLC data.
    global reverse_vocab, vocab, vocab_size, lm

    if FLAGS.lmfile is not None:
        print("Loading Language model from %s" % FLAGS.lmfile)
        lm = kenlm.LanguageModel(FLAGS.lmfile)

    print("Preparing NLC data in %s" % FLAGS.data_dir)

    x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
        FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size, tokenizer=nlc_data.char_tokenizer)
    vocab, reverse_vocab = nlc_data.initialize_vocabulary(vocab_path)
    # print(vocab)
    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)


def load_model():
    tf.reset_default_graph()
    global model, sess
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    sess = tf.InteractiveSession()
    model = create_model(sess, vocab_size, False)


def decode(sent, top_n=1):
    global model, sess
    if model is None and sess is None:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        sess = tf.InteractiveSession()
        model = create_model(sess, vocab_size, False)
    else:
        pass
    return fix_sent(model, sess, sent, top_n)


def write_prediction_result(test_path, prediction_path):
    f = open(prediction_path, 'w')
    count = 0
    with open(test_path) as file:
        for line in file.readlines():
            count += 1
            line = line.strip()
            f.write(decode(line)[0][0] + '\n')

            if count%10000 == 0:
                print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Decoding")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    # Load model and vocabulary
    load_vocab()
    load_model()

    # Decode input sentence
    write_prediction_result(args.test_path, args.output_path)
