from flask import Flask, redirect, request, render_template, url_for, send_file
from werkzeug.utils import secure_filename
import insightface
import cv2
import pandas as pd
from PIL import Image
import os
from io import StringIO
import numpy as np
import base64

app = Flask(__name__)


UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = 'UPLOAD_FOLDER/'
ALLOWED_EXTENSIONS = ["mp4", "JPG", "PNG", "GIF"]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        f = request.files['file']
        if allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            print("Image saved")
            return redirect('/UPLOAD_FOLDER/'+ filename)
    else:
        return render_template('video.html')


# @app.route('/video', methods=['POST', 'GET'])
# def video():
#     if request.method == 'POST':
#         file = request.files['file']
#         return redirect(url_for('result', file='File upload successfully.'))
#         #
#         # if file.filename == "":
#         #     print("No filename")
#         #     return render_template('video.html')
#
#         # if allowed_file(file.filename):
#         #     filename = secure_filename(file.filename)
#         #     file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
#         #     print("Image saved")
#         #     return redirect(url_for('result', file='File upload successfully.'))
#         #
#         # else:
#         #     print("That file extension is not allowed")
#         #     return render_template('video.html')

@app.route("/UPLOAD_FOLDER/<filename>", methods = ['GET'])
def download_file(filename):
    #CV2 does not like relative path
    v_cap = cv2.VideoCapture('/Users/dugongzzz/Documents/GitHub/cmpt733-deepfake-detection/website/UPLOAD_FOLDER/' + filename)
    _, frame = v_cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=-1, nms=0.4)

    bbox, landmark = model.detect(frame, threshold=0.5, scale=1.0)
    facelist = []
    for each in bbox:
        boundary = each.tolist()
        x, y, w, h = boundary[0:4]
        detected_face = frame[int(y):int(h), int(x):int(w)]
        facelist.append(detected_face)
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 255), 3)

    # In memory
    image_content = cv2.imencode('.jpg', frame)[1].tostring()
    encoded_image = base64.encodebytes(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('result.html', content=to_send)

if __name__ == "__main__":
    app.run(host='0.0.0.0')

