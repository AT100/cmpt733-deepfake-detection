from flask import Flask, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename
import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import os
import pandas as pd
from glob import glob
import matplotlib.pylab as plt
from PIL import Image
import os


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        f = request.files['file']
        return redirect(url_for('result', file='File upload successfully.'))
    else:
        return render_template('video.html')


@app.route('/result/<file>')
def result(file):
    v_cap = cv2.VideoCapture(file)
    _, frame = v_cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

    bbox, landmark = model.detect(frame, threshold=0.5, scale=1.0)
    for each in bbox:
        boundary = each.tolist()
        x, y, w, h = boundary[0:4]
        detected_face = frame[int(y):int(h), int(x):int(w)]
        return detected_face

if __name__ == "__main__":
    app.run(debug=True)
