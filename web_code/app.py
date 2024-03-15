from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
app.secret_key = "super secret key"  # Necessary for flash messages to work

model = tf.keras.models.load_model('/Users/fisayo_ojo/Documents/More_IS/model.h5')


tokenizer = Tokenizer(num_words=5000)
max_length = 100  

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if request.method == 'POST':

        document1 = request.files['document1']
        document2 = request.files['document2']
        
        if document1 and document2:
            filename1 = secure_filename(document1.filename)
            filename2 = secure_filename(document2.filename)
            temp_dir = 'temp_uploads'
            os.makedirs(temp_dir, exist_ok=True)
            filepath1 = os.path.join(temp_dir, filename1)
            filepath2 = os.path.join(temp_dir, filename2)
            document1.save(filepath1)
            document2.save(filepath2)
         
            prediction, confidence = process_files_and_predict(filepath1, filepath2)
            

            os.remove(filepath1)
            os.remove(filepath2)

            flash(f"The documents are {'related' if prediction else 'not related'} with a confidence of {confidence:.2f}")
            return redirect(url_for('index'))
        else:
            flash("Failed to upload files. Please ensure both files are selected.")
            return redirect(url_for('index'))

def process_files_and_predict(filepath1, filepath2):

    new_docs = []
    for path in [filepath1, filepath2]:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
            new_docs.append(text)
    
    new_sequences = tokenizer.texts_to_sequences(new_docs)
    new_padded_docs = pad_sequences(new_sequences, maxlen=max_length, padding='post')
    new_pair = np.array([new_padded_docs[0], new_padded_docs[1]]).reshape(1, 2, -1)
    

    new_prediction = model.predict([new_pair[:, 0], new_pair[:, 1]])
    related = new_prediction[0][0] > 0.5
    return related, new_prediction[0][0]

if __name__ == '__main__':
    app.run(debug=True)
