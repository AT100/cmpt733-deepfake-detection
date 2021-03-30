from flask import Flask, redirect, request, render_template, url_for, send_file, Response
from werkzeug.utils import secure_filename
# from camera import VideoCamera
# from pytube import YouTube
import insightface
# import logging
import cv2
import os
import base64
import boto3


app = Flask(__name__)

s3 = boto3.client('s3',
                    aws_access_key_id='AKIA4VF4SCA5INBHU4TJ',
                    aws_secret_access_key= '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'
                    #aws_session_token='secret token here'
                     )

# UPLOAD_FOLDER ='static/uploads/'
# DOWNLOAD_FOLDER = 'static/downloads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

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
            s3.upload_file(
                Bucket='nbayeah',
                Filename=filename,
                Key=filename
            )
            print("Image saved")
            return redirect('/result')
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

@app.route("/result/<filename>", methods = ['GET'])
def download_file(filename):
    #CV2 does not like relative path
    # s3.download_file(app.config['nbayeah'],
    #                  filename,
    #                  os.path.join('tmp',filename))
    response = s3.generate_presigned_url('get_object',
                                             Params={'Bucket': 'nbayeah', 'Key': filename},
                                             ExpiresIn=24)
    v_cap = cv2.VideoCapture(response)
    _, frame = v_cap.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.normalize(frame1, frame1, 0, 255, cv2.NORM_MINMAX)

    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=-1, nms=0.4)

    bbox, landmark = model.detect(frame1, threshold=0.5, scale=1.0)
    facelist = []
    for each in bbox:
        boundary = each.tolist()
        x, y, w, h = boundary[0:4]
        detected_face = frame1[int(y):int(h), int(x):int(w)]
        facelist.append(detected_face)
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 255), 3)

    # In memory
    image_content = cv2.imencode('.jpg', frame)[1].tostring()
    encoded_image = base64.encodebytes(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('result.html', content=to_send)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
