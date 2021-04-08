from flask import Flask, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename
# from pytube import YouTube
from facenet_pytorch import MTCNN
# import logging
import os
import cv2
import base64
import boto3
from PIL import Image
from google.cloud import datastore, storage
#import keras
#from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
#import h5py
# import tensorflow as tf
#from tensorflow.python.lib.io import file_io

app = Flask(__name__)

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/dugongzzz/Downloads/cmpt733-309217-e4dd118f17bf.json"

client = datastore.Client()
key = client.key('Settings', 5634161670881280)
result = client.get(key)

S3_KEY = result['s3_code']
S3_CODE = result['s3_key']
s3 = boto3.client('s3', aws_access_key_id=S3_KEY, aws_secret_access_key=S3_CODE)

storage_client = storage.Client()
bucket = storage_client.bucket('cmpt733')
blob = bucket.blob('model.h5')
blob.download_to_filename('./model.h5')
new_model = load_model('./model.h5')

ALLOWED_EXTENSIONS = ["mp4"]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == "":
            print("No filename")
            return render_template('video.html')

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            s3.upload_fileobj(
                file,
                'cmpt733sfu',
                file.filename,
                ExtraArgs={
                    "ACL": 'public-read',
                    "ContentType": file.content_type  # Set appropriate content type as per the file
                }
            )

            return redirect(url_for('result', filename=filename))
    else:
        return render_template('video.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
