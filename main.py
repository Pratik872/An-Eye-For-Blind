
from utils import *
from constants import  *
import os
from werkzeug.utils import secure_filename

# Importing necessary libraries for Flask Application
from flask import Flask,render_template,request


# Initialising flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        test_image = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(test_image.filename))
        test_image.save(file_path)

        processed_image = PreprocessImage(file_path)
        extracted_features_V3 = ExtractFeaturesV3(processed_image)
        extracted_features_encoder = ExtractFeaturesEncoder(extracted_features_V3)
        predicted_caption = PredictCaptionDecoder(extracted_features_encoder)
        pred_caption = ReturnCaption(predicted_caption)
        # SpeakOutCaption(aud_file)

    return render_template('home.html', prediction_text = pred_caption)



if __name__ == '__main__':
    app.run(debug=True)