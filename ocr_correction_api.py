#!/bin/python
# -*- coding: utf-8 -*-
from flask import request, jsonify, Flask
import os
# import sys
from util import evaluate
from decode import decode, load_vocab, load_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        ocr_text = request.get_json()
        print(ocr_text)

        original_sent = ocr_text['ocr_text']
        output_sent = decode(original_sent)[0][0]
        evals = evaluate(original_sent, output_sent)
        result = {'predicted': output_sent,
                  'evaluated': evals}

        print(result)
        return jsonify(result)
    else:
        return '<h1>Error</h1>'


if __name__ == '__main__':
    # load vocabulary and model
    load_vocab()
    load_model()

    # run app
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    app.run(port=6002, host="0.0.0.0")
