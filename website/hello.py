from flask import Flask, redirect, request, render_template, url_for, send_file, Response
from werkzeug.utils import secure_filename
# from pytube import YouTube
from facenet_pytorch import MTCNN
# import logging
#import os
import cv2
import base64
import boto3
from PIL import Image
from google.cloud import datastore
import keras
from keras.models import load_model

app = Flask(__name__)

model_url = 'https://storage.googleapis.com/cmpt733/model.h5'
new_model = load_model(model_url)

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/dugongzzz/Downloads/cmpt733-309217-e4dd118f17bf.json"
client = datastore.Client()
key = client.key('Settings', 5634161670881280)
# get by key for this entity
result = client.get(key)


S3_KEY = result['s3_code']
S3_CODE = result['s3_key']

s3 = boto3.client('s3', aws_access_key_id=S3_KEY, aws_secret_access_key=S3_CODE)

ALLOWED_EXTENSIONS = ["mp4", "JPG", "PNG"]



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


@app.route("/result/<filename>", methods = ['GET'])
def result(filename):
    response = s3.generate_presigned_url('get_object',
                                             Params={'Bucket': 'cmpt733sfu', 'Key': filename},
                                             ExpiresIn=500)
    url = 'https://cmpt733sfu.s3.amazonaws.com/' + filename
    v_cap = cv2.VideoCapture(url)
    print(response)
    _, frame = v_cap.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.normalize(frame1, frame1, 0, 255, cv2.NORM_MINMAX)

    new_frame = Image.fromarray(frame)
    mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False)  # select_largest=False, device='cuda')
    #test = mtcnn(new_frame)

    # detect faces in the image
    faces = mtcnn.detect(new_frame)

    for each in faces[0]:
        each1 = each.tolist()
        x, y, w, h = each1
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 255), 3)

    scale = 0.5
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    resized_img_with_scaling = cv2.resize(frame, (width, height))

    new_model.predict(resized_img_with_scaling)

    # In memory
    image_content = cv2.imencode('.jpg', resized_img_with_scaling)[1].tostring()
    encoded_image = base64.encodebytes(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('result.html', content=to_send)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
