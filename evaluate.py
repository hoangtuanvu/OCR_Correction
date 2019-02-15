# -*- coding:utf-8 -*-

import pickle
from decode import decode, load_model, load_vocab
from util import evaluate
import argparse
import codecs


def get_ratios(source_path, target_path, out_path=''):
    correct_ratios = []
    error_ratios = []

    same_values_in_testset = 0
    same_values_in_pred_test = 0
    same_values_in_total_testset = 0

    count = 0

    # write files
    if out_path != '':
            f = open(out_path, 'w')

    try:
        with codecs.open(source_path, encoding='utf-8') as fr1:
            with codecs.open(target_path, encoding='utf-8') as fr2:
                for sent, sentt in zip(fr1, fr2):
                    count += 1
                    if count % 100 == 0:
                        print('Model has processed {}'.format(count))

                    correct_sent = sentt.strip()
                    original_sent = sent.strip()
                    revised_sent = decode(original_sent)[0][0]

                    # Evaluates original sentence and correct sentence in testset
                    error_ratio = evaluate(original_sent, correct_sent)
                    error_ratios.append(error_ratio)
                    if error_ratio == 0.0:
                        same_values_in_testset += 1

                    # Evaluates revised sentence and correct sentence in testset
                    correct_ratio = evaluate(revised_sent, correct_sent)
                    correct_ratios.append(correct_ratio)
                    if correct_ratio == 0.0:
                        same_values_in_pred_test += 1
                        if error_ratio == 0.0:
                            same_values_in_total_testset += 1

                    # Write files
                    if out_path != "":
                        f.write(original_sent + ' ' + revised_sent + ' ' + correct_sent + ' ' + str(correct_ratio) + '\n')

        # Calculate CER, WER
        avg_error_ratio = sum(error_ratios) * 1.0 / len(error_ratios)
        avg_correct_ration = sum(correct_ratios) * 1.0 / len(correct_ratios)
        original_line_acc = same_values_in_testset * 1.0 / count
        predited_line_acc = same_values_in_pred_test * 1.0 / count

        print("average error ratio: {}".format(avg_error_ratio))
        print("average correct ratio: {}".format(avg_correct_ration))
        print(
            "Total pairs of sentences are the same between OCR test and its Ground truth: {}".format(
                same_values_in_testset))
        print("Total pairs of sentences are the same between Prediction test and its Ground truth: {}".format(
            same_values_in_pred_test))
        print("Total pairs of sentences are the same in all sets of testset: {}".format(same_values_in_total_testset))

        # Write error ratios of model
        fp = open('ratios.pkl', 'wb')
        pickle.dump([error_ratios, correct_ratios], fp)
        fp.close()

        return avg_error_ratio, avg_correct_ration, original_line_acc, predited_line_acc
    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating OCR Correction model")
    parser.add_argument("--ocr_text", required=True)
    parser.add_argument("--ground_truth_text", required=True)
    parser.add_argument("--out_path", default="")

    args = parser.parse_args()

    # Load model and vocabulary
    load_vocab()
    load_model()

    # Evaluate model
    get_ratios(args.ocr_text, args.ground_truth_text, args.out_path)
