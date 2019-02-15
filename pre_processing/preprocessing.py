import csv
from charguana import get_charset
from sklearn.model_selection import train_test_split
import os
import sys
import argparse

PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../')

from nlc_data import _START_VOCAB


# Create datasets
def create_inout_dataset(source_path, target_path, original_path):
    f_in = open(source_path, 'w')
    f_out = open(target_path, 'w')

    a = {}
    with open(original_path) as f:
        for line in f.readlines():
            tmp = line.strip().split()

            if len(tmp) == 3:
                ocr_text = line.split()[0]
                gt_text = line.split()[2]

                if (ocr_text + '|' + gt_text) not in a.keys():
                    a[ocr_text + '|' + gt_text] = 1
                    f_in.write(ocr_text + '\n')
                    f_out.write(gt_text + '\n')
                # f_in.write(ocr_text + '\n')
                # f_out.write(gt_text + '\n')


# Create vocabulary and character index
def create_general_vocab(source_path, target_path, kanji_folder, output_folder):
    def create_vocab(file_path):
        tmp = {}
        with open(file_path) as f:
            for line in f.readlines():
                line = line.strip()
                for obj in line:
                    if obj not in tmp.keys():
                        tmp[obj] = 1
                    else:
                        tmp[obj] += 1
        return tmp

    def create_database(folder):
        tmp = []
        for file in os.listdir(folder):
            with open(folder + os.sep + file, newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    tmp.append(row[0].split()[0])

        for ch in list(set(get_charset('katakana'))):
            tmp.append(ch)

        for ch in list(set(get_charset('hiragana'))):
            tmp.append(ch)

        tmp = list(set(tmp))

        return tmp

    # Create candidates list from OCR text and Ground Truth text
    candidates = {}
    candidates.update(create_vocab(source_path))
    candidates.update(create_vocab(target_path))
    candidates = sorted(candidates, key=candidates.get, reverse=True)

    # Create the most popular japanese character
    database = create_database(kanji_folder)

    # create vocab.dat
    with open(output_folder + os.sep + 'vocab.dat', 'w') as f:
        for ch in _START_VOCAB:
            f.write(ch + '\n')
        for item in candidates:
            f.write(item + '\n')
        for item in database:
            if item not in candidates:
                f.write(item + '\n')


# Create training and validating dataset
def create_training_and_valid_set(source_path, target_path, output_folder, ratio=0.2):
    def get_lines(path):
        tmp = []
        with open(path, 'r') as f:
            for line in f.readlines():
                tmp.append(line.strip())

        return tmp

    # write files
    def write_dataset(file_path, inputs):
        with open(file_path, 'w') as f:
            for item in inputs:
                f.write(item + '\n')

    # Separate original sets into training and validating sets by using sklearn lib
    X_training, X_valid, Y_training, Y_valid = train_test_split(get_lines(source_path), get_lines(target_path),
                                                                test_size=ratio)

    write_dataset(output_folder + os.sep + 'train.x.txt', X_training)
    write_dataset(output_folder + os.sep + 'valid.x.txt', X_valid)
    write_dataset(output_folder + os.sep + 'train.y.txt', Y_training)
    write_dataset(output_folder + os.sep + 'valid.y.txt', Y_valid)


def create_separated_sent(target_path, output_folder):
    f_out = open(output_folder + os.sep + 'separated_sent.txt', 'w')
    if not os.path.exists(target_path):
        print(target_path + 'does not exist')
        return

    with open(target_path, 'r') as f:
        for line in f.readlines():
            tmp = ' '.join([item for item in line.strip()])
            f_out.write(tmp + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process dataset before training, validating and testing OCR Correction Model")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--target_path", required=True)
    # parser.add_argument("--original_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--kanji_folder", required=True)
    parser.add_argument("--ratio", type=float, default=0.2)

    args = parser.parse_args()
    # create_inout_dataset(args.source_path, args.target_path, args.original_path)
    create_general_vocab(args.source_path, args.target_path, args.kanji_folder, args.output_folder)
    create_training_and_valid_set(args.source_path, args.target_path, args.output_folder, args.ratio)
    create_separated_sent(args.target_path, args.output_folder)

# create_inout_dataset('/home/brian/Downloads/name_??ackaged/417/x.txt', '/home/brian/Downloads/name_packaged/417/y.txt', '/home/brian/Downloads/name_packaged/417/test_417.txt')
