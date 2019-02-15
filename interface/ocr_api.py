import requests
import os
import csv
import argparse
import sys


# Depends on csv file format
def read_csv_file(path):
    """

    :param path: file path
    :return: dictionary with key = image name, value = grouth truth text respectively
    """
    name_to_labels = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        for col in reader:
            tmp = col[6].split(':')
            if len(tmp) == 2:
                name_to_labels[col[0]] = tmp[1].split('"')[1]

    return name_to_labels


# Check if file is existed or not
def file_existence(file_path):
    if os.path.exists(file_path):
        return True

    return False


def processor(out_src_path, out_target_path, csv_file_path, images_folder, api_url):
    """

    :param out_src_path: ocr text path
    :param out_target_path: grouth truth path
    :param csv_file_path:
    :param images_folder: contains list of images for predicting through OCR model
    :param api_url: OCR API link
    """
    try:
        # Gets pairs of sentences
        name_to_labels = read_csv_file(csv_file_path)
        f_ocr = open(out_src_path, 'w')
        f_gt = open(out_target_path, 'w')

        count = 0
        for file in os.listdir(images_folder):
            if file.endswith('jpg'):
                image = open(images_folder + os.sep + file, "rb").read()
                payload = {"image": image}

                # submit the request
                res = requests.post(api_url, files=payload).json()

                # ensure the request was successful
                count += 1
                if res["success"]:
                    pred = res["prediction"]
                    print(pred)
                    if '_pad_' in pred:
                        pred = pred.replace('_pad_', '')
                    f_ocr.write(pred + '\n')
                    f_gt.write(name_to_labels[file] + '\n')
                else:
                    print("Request failed")
    except ValueError:
        print('Error when processing OCR through API')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Processing through OCR API")
    parser.add_argument("--api_url", default="http://65.49.54.195:6001/process?type=1")
    parser.add_argument("--ocr_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--csv_file_path", required=True)
    parser.add_argument("--images_folder", required=True)

    args = parser.parse_args()

    if not file_existence(args.csv_file_path) or not file_existence(args.images_folder):
        print('File or Folder are not existed')
        sys.exit(0)

    processor(args.ocr_path, args.gt_path, args.csv_file_path, args.images_folder, args.api_url)

