from flask import Flask, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


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
