import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Create Flask App
app = Flask(__name__)

# Limit content size
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(filepath):
    audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast', duration=5)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1,-1,80)


# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results', filename=filename))
    return render_template("index.html")

# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    print("inside the function")
    filename = request.args['filename']
    # Compute audio signal features
    features = extract_features(filename)
    # Load model and perform inference
    model = load_model('model')
    prediction = model.predict(features)
    labelencoder = LabelEncoder()
    labelencoder.fit_transform(['Airplane', 'Bics', 'Cars', 'Helicopter', 'Motocycles', 'Train', 'Truck', 'bus'])
    classes_x = np.argmax(prediction, axis=1)
    prediction_class = labelencoder.inverse_transform(classes_x)
    prediction_class
    # Delete uploaded file
    os.remove(filename)
    # Render results
    return render_template("results.html", filename=filename, prediction_to_render=prediction_class[0])


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
