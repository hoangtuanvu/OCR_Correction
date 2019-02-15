from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import numpy as np
import tensorflow as tf
import nlc_data
import nlc_model
from util import pair_iter
from util import write_anomaly
import pickle
import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.00015, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.3, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 50, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 5000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 100, "Maximum sequence length.")
tf.app.flags.DEFINE_float("anomaly_threshold", 0.2, "Anomaly Threshold.")
tf.app.flags.DEFINE_integer("anomaly_epochs", 20, "Anomaly Epochs")
tf.app.flags.DEFINE_string("data_dir", "./data_dir/data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./data_dir/train_data", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "CHAR / WORD.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
FLAGS = tf.app.flags.FLAGS

SOURCE_PATH = FLAGS.data_dir + os.sep + FLAGS.tokenizer.lower() + os.sep + 'anomaly/anomaly_x.txt'
TARGET_PATH = FLAGS.data_dir + os.sep + FLAGS.tokenizer.lower() + os.sep + 'anomaly/anomaly_y.txt'


def create_model(session, vocab_size, forward_only):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only, optimizer=FLAGS.optimizer)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def validate(model, sess, x_dev, y_dev):
    cost_all = 0
    step = 0
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, FLAGS.batch_size,
                                                                            FLAGS.num_layers):
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        lengths = np.sum(target_mask, axis=0)
        mean_length = np.mean(lengths)
        cost = cost / mean_length
        cost_all += cost
        step += 1
    return cost_all / step


def train():
    """Train a translation model using NLC data."""
    # Prepare NLC data.
    logging.info("Preparing NLC data in %s" % FLAGS.data_dir)
    x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
        FLAGS.data_dir + os.sep + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size, tokenizer=nlc_data.char_tokenizer)
    vocab, _ = nlc_data.initialize_vocabulary(vocab_path)
    vocab_size = len(vocab)
    logging.info("Vocabulary size: %d" % vocab_size)

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, vocab_size, False)

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        print("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        epoch = 0
        best_epoch = 0
        train_costs = []
        valid_costs = []
        previous_valid_losses = []
        while FLAGS.epochs == 0 or epoch < FLAGS.epochs:
            epoch += 1
            current_step = 0
            epoch_cost = 0
            epoch_tic = time.time()
            for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, FLAGS.batch_size,
                                                                                    FLAGS.num_layers):
                # Get a batch and make a step.fa
                grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

                lengths = np.sum(target_mask, axis=0)
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)

                cost = cost / mean_length
                epoch_cost += cost
                current_step += 1

                if current_step % FLAGS.print_every == 0:
                    logging.info('epoch %d, iter %d, cost %f, length mean/std %f/%f' % (
                        epoch, current_step, cost, mean_length, std_length))

                    if (epoch >= FLAGS.anomaly_epochs) and \
                            (cost >= FLAGS.anomaly_threshold):
                        write_anomaly(source_tokens, vocab_path,
                                      SOURCE_PATH + '_' + str(epoch) + '_' + str(current_step))
                        write_anomaly(target_tokens, vocab_path,
                                      TARGET_PATH + '_' + str(epoch) + '_' + str(current_step))

            # One epoch average train cost
            train_costs.append(epoch_cost / current_step)

            # After one epoch average validate cost
            epoch_toc = time.time()
            epoch_time = epoch_toc - epoch_tic
            valid_cost = validate(model, sess, x_dev, y_dev)
            valid_costs.append(valid_cost)
            logging.info("Epoch %d Validation cost: %f time:to %2fs" % (epoch, valid_cost, epoch_time))

            # Checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")
            if len(previous_valid_losses) > 2 and valid_cost > previous_valid_losses[-1]:
                logging.info("Annealing learning rate by %f" % FLAGS.learning_rate_decay_factor)
                sess.run(model.learning_rate_decay_op)
                model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_valid_losses.append(valid_cost)
                best_epoch = epoch
                model.saver.save(sess, checkpoint_path, global_step=epoch)

        pickle.dump([train_costs, valid_costs], open('costs_data.pkl', 'wb'))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
