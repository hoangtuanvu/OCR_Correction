from util import remove_same_pairs
from charguana import get_charset
import csv
import os
import random


def get_kanji_vocab(folder):
    tmp = []
    for file in os.listdir(folder):
        with open(folder + os.sep + file, newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                tmp.append(row[0].split()[0])
    tmp = list(set(tmp))
    return tmp


def get_hira_vocab():
    tmp = []
    for ch in list(set(get_charset('hiragana'))):
        tmp.append(ch)
    tmp = list(set(tmp))
    return tmp


def get_kata_vocab():
    tmp = []
    for ch in list(set(get_charset('katakana'))):
        tmp.append(ch)

    tmp = list(set(tmp))
    return tmp


def get_dict(source_path, target_path):
    a = {}
    with open(source_path, 'r') as f1:
        with open(target_path, 'r') as f2:
            for (sent1, sent2) in zip(f1, f2):
                sent1 = sent1.strip()
                sent2 = sent2.strip()
                a[sent1 + '|' + sent2] = 1
    return a


def get_content_csv_files(path, delimiter=','):
    a = []
    count = 0
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter, quotechar='|')
        for row in reader:
            count += 1
            if count == 1:
                continue
            a.append(row[1])
    a = list(set(a))
    return a


def remove_randomly(source_path, target_path, src_out, tag_out):
    a = {}

    f_ocr = open(src_out, 'w')
    f_gt = open(tag_out, 'w')

    with open(source_path) as f1:
        with open(target_path) as f2:
            for (sent1, sent2) in zip(f1, f2):
                sent1 = sent1.strip()
                sent2 = sent2.strip()

                if sent2 not in a:
                    a[sent2] = [sent1]
                else:
                    tmp = a[sent2].copy()
                    tmp.append(sent1)
                    a[sent2] = tmp
    print(len(a))

    for item in a.keys():
        values = a[item]
        f_ocr.write(random.choice(values) + '\n')
        f_gt.write(item + '\n')


def preprocess():
    remove_same_pairs('/home/brian/Downloads/name_dataset_8/x.txt',
                      '/home/brian/Downloads/name_dataset_8/y.txt',
                      '/home/brian/Downloads/name_dataset_8/x.txt_',
                      '/home/brian/Downloads/name_dataset_8/y.txt_')

    a = get_dict('/home/brian/Downloads/name_dataset_8/x.txt_', '/home/brian/Downloads/name_dataset_8/y.txt_')
    b = get_dict('/home/brian/Downloads/name_packaged/417/x.txt', '/home/brian/Downloads/name_packaged/417/y.txt')
    c = list(set(a) & set(b))
    print(len(c))
    f_ocr = open('/home/brian/Downloads/name_dataset_8/x.txt', 'w')
    f_gt = open('/home/brian/Downloads/name_dataset_8/y.txt', 'w')

    for item in a:
        if item not in b:
            tmp = item.split('|')
            f_ocr.write(tmp[0] + '\n')
            f_gt.write(tmp[1] + '\n')


from nltk import edit_distance


def evaluate(sentx, senty):
    sent_max_len = max(len(list(sentx)), len(list(senty)))
    if sent_max_len == 0:
        return 0

    return edit_distance(sentx, senty) / sent_max_len


if __name__ == "__main__":
    # count = 0
    # error_ratio = 0.0
    # with open('/home/brian/Downloads/name_packaged/Frequency/x.txt_') as f1:
    #     with open('/home/brian/Downloads/name_packaged/Frequency/y.txt_') as f2:
    #         for (sent1, sent2) in zip(f1, f2):
    #             sent1 = sent1.strip()
    #             sent2 = sent2.strip()
    #             count += 1
    #
    #             error_ratio += evaluate(sent1, sent2)
    #
    # print(error_ratio / count)

    preprocess()
    # a = {}
    # for item in get_kata_vocab():
    #     a[item] = 1
    #
    # for item in get_hira_vocab():
    #     a[item] = 1
    #
    # f_out = open('/home/brian/Downloads/name_packaged/Frequency/last_names.txt_', 'w')
    # b = {}
    # with open('/home/brian/Downloads/name_packaged/Frequency/last_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         tmp = line.split(',')
    #
    #         if len(tmp) == 5:
    #             is_checked = True
    #             for item in tmp[1]:
    #                 if item in a:
    #                     is_checked = False
    #                     break
    #             if is_checked:
    #                 if tmp[1] not in b:
    #                     b[tmp[1]] = 1
    #                     f_out.write(tmp[1] + '\n')
    # a = []
    # with open('/home/brian/Downloads/name_packaged/Frequency/Freq_Last.txt') as f:
    #     for line in f.readlines():
    #         a.append(line.strip())
    #
    # b = []
    # with open('/home/brian/Downloads/name_packaged/Frequency/First.txt') as f:
    #     for line in f.readlines():
    #         b.append(line.strip())
    #
    # f_out = open('/home/brian/Downloads/name_packaged/Frequency/full_names_2.txt', 'w')
    # for i in range(len(a)):
    #     for j in range(len(b)):
    #         f_out.write(a[i] + b[j] + '\n')

    # a = {}
    # with open('/home/brian/Downloads/name_dataset_10/kanji_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             a[line] = 1
    #
    # f_out = open('/home/brian/Downloads/name_packaged/Frequency/full.txt', 'w')
    # with open('/home/brian/Downloads/name_packaged/Frequency/full_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             # a[line] = 1
    #             f_out.write(line + '\n')

    # count = 0
    # with open('/home/brian/Downloads/name_dataset_10/x.txt') as f1:
    #     with open('/home/brian/Downloads/name_dataset_10/y.txt') as f2:
    #         for (sent1, sent2) in zip(f1, f2):
    #             sent1 = sent1.strip()
    #             sent2 = sent2.strip()
    #             error_ratio = edit_distance(sent1, sent2)
    #             if error_ratio > 0 and len(sent1) == len(sent2):
    #                 count += 1
    #
    #                 if count < 20:
    #                     print(sent1 + "|" + sent2)
    # print(count)


    # chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    #          'v', 'w', 'x', 'y', 'z']
    # numbs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    #
    # a = {}
    # for ch in chars:
    #     a[ch] = 1
    #     a[ch.upper()] = 1
    # for numb in numbs:
    #     a[numb] = 1
    #
    # f_out = open('/home/brian/Cinnamon/Name_Dataset/real_names/JMnedict_2.txt', 'w')
    # with open('/home/brian/Cinnamon/Name_Dataset/real_names/JMnedict.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #
    #         new_it = line[5:len(line)-6]
    #
    #         cp = unicodedata.normalize('NFKD', new_it)
    #         is_passed = True
    #         for ch in cp:
    #             if ch in a:
    #                 is_passed = False
    #                 break
    #
    #         if is_passed:
    #             f_out.write( new_it + '\n')

    # f_out = open('/home/brian/Cinnamon/Name_Dataset/real_names/JMnedict_3.txt', 'w')
    #
    # a = {}
    # with open('/home/brian/Downloads/name_dataset_8/y.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             a[line] = 1
    #
    # with open('/home/brian/Cinnamon/Name_Dataset/real_names/JMnedict_2.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             f_out.write(line + '\n')

    a = {}
    # #
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    numbs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    for ch in chars:
        a[ch] = 1
        a[ch.upper()] = 1
    for numb in numbs:
        a[numb] = 1
    # # # # for item in get_kanji_vocab('/home/brian/Downloads/kanji_characters'):
    # # # #     a[item] = 1
    # # # #
    # # # for item in get_kata_vocab():
    # # #     a[item] = 1
    # # #
    # # # for item in get_hira_vocab():
    # # #     a[item] = 1
    # # # #
    f_ocr = open('/home/brian/Downloads/name_dataset_8/x.txt_', 'w')
    f_gt = open('/home/brian/Downloads/name_dataset_8/y.txt_', 'w')

    f = open('/home/brian/Downloads/name_dataset_8/XXX.txt', 'w')
    with open('/home/brian/Downloads/name_dataset_8/x.txt') as f1:
        with open('/home/brian/Downloads/name_dataset_8/y.txt') as f2:
            for (sent1, sent2) in zip(f1, f2):
                sent1 = sent1.strip()
                sent2 = sent2.strip()

                is_existed = False
                for item in sent2:
                    if item in a:
                        is_existed = True
                        break
                if not is_existed:
                    f_ocr.write(sent1 + '\n')
                    f_gt.write(sent2 + '\n')
                else:
                    f.write(sent2 + '\n')
    # a = {}
    # with open('/home/brian/Cinnamon/Name_Dataset/SyntheticData/name_full_kanji_katakana.txt') as f:
    #     for line in f.readlines():
    #         a[line.strip()] = 1
    #
    # b = {}
    # with open('/home/brian/Downloads/name_dataset_8/y.txt') as f:
    #     for line in f.readlines():
    #         if line.strip() not in b:
    #             b[line.strip()] = 1
    #
    # c = list(set(a.keys()) & set(b.keys()))
    # print(len(c))

    #
    # with open('/home/brian/Downloads/name_dataset_8/x.txt') as f1:
    #     with open('/home/brian/Downloads/name_dataset_8/y.txt') as f2:
    #         for (sent1, sent2) in zip(f1, f2):
    #             sent1 = sent1.strip()
    #             sent2 = sent2.strip()
    #             if sent2 not in a:
    #                 f_ocr.write(sent1 + '\n')
    #                 f_gt.write(sent2 + '\n')

    # remove_randomly('/home/brian/Downloads/name_dataset_8/x.txt',
    #                 '/home/brian/Downloads/name_dataset_8/y.txt',
    #                 '/home/brian/Downloads/name_dataset_8/x.txt_',
    #                 '/home/brian/Downloads/name_dataset_8/y.txt_')
    # a = {}
    # f_ocr = open('/home/brian/Downloads/name_dataset_8/x.txt', 'w')
    # f_gt = open('/home/brian/Downloads/name_dataset_8/y.txt', 'w')
    #
    # with open('/home/brian/Downloads/name_dataset_8/x.txt_') as f1:
    #     with open('/home/brian/Downloads/name_dataset_8/y.txt_') as f2:
    #         for (sent1, sent2) in zip(f1, f2):
    #             sent1 = sent1.strip()
    #             sent2 = sent2.strip()
    #             if sent1 not in a:
    #                 a[sent1] = 1
    #                 f_ocr.write(sent1 + '\n')
    #                 f_gt.write(sent2 + '\n')
    #
    # print(len(a))

    # a = {}
    # with open('/home/brian/Downloads/name_dataset_9/out_6.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         name = line.split()[2]
    #         if name not in a:
    #             a[name] = 1
    #
    # f_out = open('/home/brian/Cinnamon/Name_Dataset/hira_kata_names/ErikDao/kata_names_1.txt', 'w')
    # with open('/home/brian/Cinnamon/Name_Dataset/hira_kata_names/ErikDao/kata_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             f_out.write(line + '\n')

    # f_out = open('/home/brian/Cinnamon/Name_Dataset/wiki_names/hira_names.txt', 'w')
    #
    # a = {}
    # count = 0
    # with open('/home/brian/Cinnamon/Name_Dataset/wiki_names/new_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         tmp = line.split()
    #
    #         if tmp[2] not in a:
    #             a[tmp[2]] = 1
    #
    #         if len(tmp) == 4 and tmp[3] not in a:
    #             a[tmp[3]] = 1
    #
    #         if len(tmp) == 4 and (tmp[2] + tmp[3]) not in a:
    #             a[tmp[2] + tmp[3]] = 1
    #
    # for item in a:
    #     f_out.write(item + '\n')

    # import jaconv
    # f_out = open('/home/brian/Cinnamon/Name_Dataset/wiki_names/kata_names.txt', 'w')
    # with open('/home/brian/Cinnamon/Name_Dataset/wiki_names/hira_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         kata_name = jaconv.hira2kata(line)
    #         f_out.write(kata_name + '\n')

    # a = {}
    # with open('/home/brian/Cinnamon/Name_Dataset/wiki_names/name.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             a[line] = 1
    # b = {}
    # with open('/home/brian/Downloads/name_dataset_8/6/y.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in b:
    #             b[line] = 1
    # c = {}
    #
    # f_out = open('/home/brian/Cinnamon/Name_Dataset/wiki_names/kanji_names.txt', 'w')
    # for item in a:
    #     if item not in b:
    #         f_out.write(item + '\n')

    # real_vocab = {}
    # with open('/home/brian/Downloads/name_packaged/417/gt_kanji.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         for item in line:
    #             if item not in real_vocab:
    #                 real_vocab[item] = 1
    # print(len(real_vocab))
    #
    # a = {}
    # with open('/home/brian/Downloads/name_dataset_9/remain_kanji_name.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             a[line] = 1
    #
    # # # for item in get_kata_vocab():
    # # #     a[item] = 1
    # # #
    # # # for item in get_hira_vocab():
    # # #     a[item] = 1
    # # #
    # f_out = open('/home/brian/Downloads/name_dataset_9/remain_kanji_name_2.txt', 'w')
    # with open('/home/brian/Downloads/name_dataset_9/kanji_name.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         # is_existed = False
    #         # for item in line:
    #         #     if item not in a:
    #         #         is_existed = True
    #         #         break
    #
    #         if line not in a:
    #             f_out.write(line + '\n')

    # a = {}
    # with open('/home/brian/Cinnamon/Name_Dataset/real_names/JMnedict_2.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             a[line] = 1
    #
    # b = {}
    # with open('/home/brian/Downloads/name_dataset_8/y.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in b:
    #             b[line] = 1

    # c = list(set(a.keys()) & (set(b.keys())))
    # print(len(c))

    # f_out = open('/home/brian/Downloads/name_dataset_10/kanji_names_2.txt', 'w')
    # # with open('/home/brian/Downloads/name_dataset_10/name_full_kanji_katakana.txt')
    #
    # with open('/home/brian/Downloads/name_dataset_10/y.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line not in a:
    #             a[line] = 1
    #
    # with open('/home/brian/Downloads/name_dataset_10/kanji_names.txt') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         # is_existed = False
    #         # for item in line:
    #         #     if item in a:
    #         #         is_existed = True
    #         #         break
    #         # if not is_existed:
    #         #     f_out.write(line + '\n')
    #         if line not in a:
    #             f_out.write(line + '\n')

    # f_ocr = open('/home/brian/Downloads/name_dataset_10/x1.txt', 'w')
    # f_gt = open('/home/brian/Downloads/name_dataset_10/y1.txt', 'w')
    # f_same = open('/home/brian/Downloads/name_dataset_10/same.txt', 'w')
    #
    # with open('/home/brian/Downloads/name_dataset_10/x.txt') as f1:
    #     with open('/home/brian/Downloads/name_dataset_10/y.txt') as f2:
    #         for (sent1, sent2) in zip(f1, f2):
    #             sent1 = sent1.strip()
    #             sent2 = sent2.strip()
    #
    #             if sent1 != sent2:
    #                 f_ocr.write(sent1 + '\n')
    #                 f_gt.write(sent2 + '\n')
    #             else:
    #                 f_same.write(sent1 + '\n')

    # f_ocr = open('/home/brian/Downloads/name_dataset_8/2/x.txt_', 'w')
    # f_gt = open('/home/brian/Downloads/name_dataset_8/2/y.txt_', 'w')
    # count = 0
    # a = {}
    # with open('/home/brian/Downloads/name_dataset_8/2/x.txt') as f1:
    #     with open('/home/brian/Downloads/name_dataset_8/2/y.txt') as f2:
    #         for (sent1, sent2) in zip(f1, f2):
    #             sent1 = sent1.strip()
    #             sent2 = sent2.strip()
    #
    #             # if len(sent1) != len(sent2):
    #             #     count += 1
    #             #     if count < 10:
    #             #         print(sent1 + '          ' + sent2)
    #
    #             if len(sent1) - len(sent2) == 1 and edit_distance(sent1, sent2) == 1:
    #                 # count += 1
    #                 # if count < 20:
    #                 #     print(sent1 + '     ' + sent2)
    #
    #                 # for item in sent2:
    #                 #     if item not in sent1:
    #                 #         if item not in a:
    #                 #             a[item] = [""]
    #                 #         else:
    #                 #             a[item].append("")
    #                 # f_ocr.write(sent1 + '\n')
    #                 # f_gt.write(sent2 + '\n')
    #                 is_existed = True
    #                 for item in sent1:
    #                     if item not in sent2:
    #                         is_existed = False
    #                         break
    #                 if is_existed:
    #
    #
    # # print(len(a.keys()))
    # print(count)
