from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
app.secret_key = "super secret key"  # Necessary for flash messages to work

model = tf.keras.models.load_model('/Users/fisayo_ojo/Documents/More_IS/model.h5')


tokenizer = Tokenizer(num_words=5000)
max_length = 100  

def preprocess_text(text):
    """Remove commas, full stops, and other punctuation."""
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punctuation = "".join([char for char in text if char not in punctuation])
    return no_punctuation

def read_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return preprocess_text(full_text)

@app.route('/')
def index():

    return render_template('index.html')

def process_files_and_predict(filepath1, filepath2):
    print("Starting file processing...")
    new_docs = []
    for path in [filepath1, filepath2]:
        print(f"Reading file: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
                new_docs.append(text)
        except Exception as e:
            print(f"Failed to read file {path}: {e}")
            raise e
    
    print("Preprocessing texts...")
    new_sequences = tokenizer.texts_to_sequences(new_docs)
    new_padded_docs = pad_sequences(new_sequences, maxlen=max_length, padding='post')
    new_pair = np.array([new_padded_docs[0], new_padded_docs[1]]).reshape(1, 2, -1)
    
    print("Making prediction...")
    new_prediction = model.predict([new_pair[:, 0], new_pair[:, 1]])
    related = new_prediction[0][0] > 0.5
    print(f"Prediction complete: {'related' if related else 'not related'} with confidence {new_prediction[0][0]:.2f}")
    return related, new_prediction[0][0]

@app.route('/upload', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        document1 = request.files['document1']
        document2 = request.files['document2']
        
        if document1 and document2:
            texts = []
            for document in [document1, document2]:
                temp_pdf_path = os.path.join('temp_uploads', secure_filename(document.filename))
                document.save(temp_pdf_path)
                
                text = read_pdf(temp_pdf_path)
                texts.append(text)
                
                os.remove(temp_pdf_path)
            prediction, confidence = process_files_and_predict(texts[0], texts[1])
            flash(f"The documents are {'related' if prediction else 'not related'} with a confidence of {confidence:.2f}")
            return redirect(url_for('index'))
        else:
            flash("Failed to upload files. Please ensure both files are selected.")
            return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
