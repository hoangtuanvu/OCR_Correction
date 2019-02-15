import os
from flask import Flask, render_template, request

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        file_name = file.filename
        print('{} is the file name'.format(file_name))
        destination = "/".join([target, file_name])
        file.save(destination)

        # OCR Processing
        without_ocr_correction = ""

        # OCR Correction
        with_ocr_correcion = ""

    return render_template("upload.html", image_name=file_name, result_with_out=without_ocr_correction,
                           result_with=with_ocr_correcion)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
