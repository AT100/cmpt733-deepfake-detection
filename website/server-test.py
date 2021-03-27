from flask import Flask, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

def ValuePredictor(to_predict_list):
    v_cap = cv2.VideoCapture(videopaths[i])
    _, frame = v_cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
    new_frame = Image.fromarray(frame)
    
    bbox, landmark = model.detect(frame, threshold=0.5, scale=1.0)
    for each in bbox:
        if labeling_dict[name] == 'FAKE':
                boundary = each.tolist()
                x, y, w, h  = boundary[0:4]
                detected_face = frame[int(y):int(h), int(x):int(w)]
                plt.imshow(detected_face)
                savepath = CURR_DIR + '/dataset/train_images/fake/img' + num + '.jpg'
                plt.imsave(savepath, detected_face)

@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        #f.save(secure_filename(f.filename))
        return redirect(url_for('result', file='File upload successfully.'))
    else:
        return render_template('video.html')


@app.route('/result/<file>')
def result(file):
    return render_template('result.html', content=file)


if __name__ == "__main__":
    app.run(debug=True)
