from flask import Flask, redirect, request, render_template, url_for, send_file, Response
from werkzeug.utils import secure_filename
from camera import VideoCamera
from pytube import YouTube
import insightface
import logging
import cv2
import os
import base64
#import boto3


logger = logging.Logger('catch_all')

app = Flask(__name__)


UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

ALLOWED_EXTENSIONS = ["mp4", "JPG", "PNG", "GIF"]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# s3 = boto3.client('s3',
#                     aws_access_key_id='AKIA4VF4SCA5FRN2XZOI',
#                     aws_secret_access_key= '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'
#                     #aws_session_token='secret token here'
#                      )
# BUCKET_NAME='nbayeah'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        f = request.files['file']
        if allowed_file(f.filename):
            filename = secure_filename(f.filename)
            #f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # s3.upload_file(
            #     Bucket='nbayeah',
            #     Filename=filename,
            #     Key=filename
            # )
            print("Image saved")
            return redirect('/result')
            #return redirect('/UPLOAD_FOLDER/'+ filename)
    else:
        return render_template('video.html')

@app.route('/youtube', methods=['Get','POST'])
def youtube():

    if request.method == 'POST':
        url = request.form['url']
        print(url)

        r = 'Download Success'
        try:
            YouTube(url).streams.first().download('Youtube')
        except Exception as e:
            r = 'Download failed, please check the youtube url.'
            print('Failed to upload to ftp: ' + str(e))
            print(r)
            return redirect(url_for('result', file=r))
        return redirect(url_for('result', file=r))

    else:
        return render_template('youtube.html')


def gen(camera):
    global g_frame
    while True:
        frame = camera.get_frame()
        if frame is not None:
            g_frame = frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/stream_show', methods=['POST', 'GET'])
def stream_show():
    count = 0
    if request.method == 'POST':
        """print(type(g_frame))
        img = cv2.imdecode(g_frame, flags=1)
        cv2.imwrite('%d.jpg' % count, img)
        count += 1"""
    return render_template('stream.html')


@app.route('/stream', methods=['POST', 'GET'])
def stream():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result/<file>')
def result(file):
    return render_template('result.html', content=file)

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
    v_cap = cv2.VideoCapture(os.path.join(app.config["UPLOAD_FOLDER"], filename))
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
