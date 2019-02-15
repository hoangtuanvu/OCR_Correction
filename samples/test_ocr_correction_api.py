import argparse
import requests


def processor(api_url, ocr_text):
    try:
        # create key-value data
        data = {"ocr_text": ocr_text}
        print("INPUT DATA: {}".format(data))

        # submit the request
        res = requests.post(api_url, json=data).json()
        predicted = res['predicted']
        evaluated = res['evaluated']
        print("OCR_CORRECTION_TEXT: {}, EDIT_DISTANCE: {}".format(predicted, evaluated))
    except ValueError:
        print('Error when processing OCR through API')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing OCR Correction API")
    parser.add_argument("--api_url", default="http://65.49.54.195:6002/predict?type=1")
    parser.add_argument("--ocr_text", required=True)

    args = parser.parse_args()
    processor(args.api_url, args.ocr_text)
