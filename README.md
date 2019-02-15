# Introduction

Implementation of Neural Language Correction (http://arxiv.org/abs/1603.09727) on Tensorflow.
But this is a Japanese version!!!

# Pre-processing data

    $ python preprocessing.py --source_path="/path/to/create/ocr_text/x.txt" --target_path="/path/to/create/ground_truth_text/y.txt" --output_folder="/dir/to/save/model_dataset" --kanji_folder="/dir/to/kanji_characters" --original_path="/path/to/pairs_of_sentence/pairs.txt" --ratio=0.2

# Generating pairs of fake dataset (Optional)

    $ python generate_fake_data.py --ocr_path_1="/path/to/ocr_path_1/ocr_path_1.txt" --ground_truth_path_1="/path/to/ground_truth_path_1/ground_truth_path_1.txt" --ocr_path_2="/path/to/ocr_path_2/ocr_path_2.txt" --ground_truth_path_2="/path/to/ground_truth_path_2/ground_truth_path_2.txt" --ocr_output_path="/path/to/save/ocr_output_text/ocr.txt" --ground_truth_output_path="/path/to/save/ground_truth_text/gt.txt"
    
# Generating binary files by using kenlm toolkit

    $/path/to/kenlm_toolkit/kenlm/build/bin/lmplz -o N < /path/to/separated_sentence_path/separated_sent.txt /path/to/save/arpa_file/corpus.arpa
    or /path/to/kenlm_toolkit/kenlm/build/bin/lmplz -o N --discount_fallback < /path/to/separated_sentence_path/separated_sent.txt > /path/to/save/arpa_file/corpus.arpa
    $/path/to/kenlm_toolkit/kenlm/build/bin/build_binary -s /path/to/arpa_file/corpus.arpa /path/to/save/binary_file/corpus.binary

**NOTICE:** You can replace N by 3, 4, 5, ... Separated sentence is derived from ground_truth.txt file (install **kenlm** python packages to load Language model and use it. You can read this [kenlm toolkits build and python package install ](https://github.com/kpu/kenlm)). 

# Training

To train character level model (for japanese, we'd better only use this mode <japanese character>):

    $ python train.py

**NOTICE:** We use Tensorflow FLAGS for customizing hyper-parameters (e.g, learning rate, dropout, batchsize, optimization functions, ...). If you want to train with another parameters, do your suitable adjustment.

# Local Decoding

    $ python decode.py --test_path="/path/to/test_path/test.txt" --output_path="/path/to/save/result.txt"
   
**NOTICE:** test.txt contains ocr output text (line-by-line) and result.txt contains desired lines respectively.

# Gets OCR output text through OCR API

    # python3 OCR_API.py --ocr_path="/path/to/save/ocr_output/x.txt" --gt_path="/path/to/save/gt_output/y.txt" --csv_file_path="/path/to/csv_file/file.csv" --images_folder="/dir/to/images"

**NOTICE:** file.csv contains list of ground truth of each image in images folder

# Web Decoding
    
    $ export FLASK_APP=server.py; flask run --host=0.0.0.0

    Now, you can post a json-format data `{'original_text':'xxx'}` to url `http://server_ip:5000/revise`, get the revised sentence.

# Tensorflow Dependency

- Tensorflow 1.4.0 and more