from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
import numpy as np
import os

output_txt_paths = [
    'articles_text/1intro.txt',
    'articles_text/2relation.txt',
    'articles_text/3count.txt',
    'articles_text/4probability.txt',
    'articles_text/libya.txt',
    'articles_text/Homework_3_Fisayo_Ojo.txt',
    'articles_text/An Investigation of the Pattern And Environmental Impact of Oil.txt',
    'articles_text/Analysis of oil spill impacts along pipelines..txt',
    'articles_text/Causes and Terrain of Oil Spillage in Niger Delta.txt',
    'articles_text/Deficient legislation sanctioning oil spill.txt',
    'articles_text/Effects of Oil Spillage (Pollution) on Agricultural Production in Delta.txt',  
    'articles_text/Effects of oil spills on fish production in the Niger Delta.txt',  
    'articles_text/EFFECTS OF OIL SPILLAGE ON FISH IN NIGERIA.txt', 
    'articles_text/Effects of oil spills on fish production in the Niger Delta.txt',  
    'articles_text/Environmental Consequences of Oil Spills on Marine Habitats and the Mitigating Measures—The Niger Delta Perspective.txt',  
    'articles_text/ENVIRONMENTAL IMPACTS OF OIL EXPLORATION.txt', 
    'articles_text/Evaluation of the Impacts of Oil Spill Disaster on Communities in Niger Delta, Nigeria.txt',  
    'articles_text/Impacts and Management of Oil Spill Pollution along the Nigerian Coastal Areas.txt',  
    'articles_text/Impacts of Oil Exploration (Oil and Gas Conflicts; Niger Delta as a Case Study).txt',  
    'articles_text/Impacts of Oil Production on Nigeria‘s Waters.txt', 
    'articles_text/NIGERIA OIL POLLUTION, POLITICS AND POLICY.txt',  
    'articles_text/Oil Pollution in Nigeria and the Issue of Human Rights of the Victims.txt', 
    'articles_text/Oil Spills and Human Health.txt',  
    'articles_text/OIL SPILLS IN THE NIGER DELTA.txt',  
    'articles_text/Press Coverage of Environmental Pollution In The Niger Delta Region of Nigeria.txt',  
    'articles_text/Shell will sell big piece of its Nigeria oil business, but activists want pollution cleaned up first _ AP News.txt'  
]

documents = []

for path in output_txt_paths:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
            documents.append(text)
    except FileNotFoundError:
        print(f"File {path} not found. Skipping.")

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(documents)

sequences = tokenizer.texts_to_sequences(documents)
max_length = max(len(x) for x in sequences)
padded_docs = pad_sequences(sequences, maxlen=max_length, padding='post')

labels = [1 if "oil" in doc.lower() else 0 for doc in documents]

pairs = []
pair_labels = []

for i in range(len(padded_docs)-1):
    for j in range(i+1, len(padded_docs)):
        pairs.append([padded_docs[i], padded_docs[j]])
        pair_labels.append(int(labels[i] == labels[j]))

pairs = np.array(pairs)
pair_labels = np.array(pair_labels)

def create_rnn_model():
    input_1 = Input(shape=(max_length,))
    input_2 = Input(shape=(max_length,))
    
    embedding = Embedding(5000, 50)  
    embedded_1 = embedding(input_1)
    embedded_2 = embedding(input_2)
    
    lstm1 = LSTM(64)(embedded_1)  
    lstm2 = LSTM(64)(embedded_2)
    
    merged = concatenate([lstm1, lstm2])
   
    dense1 = Dense(10, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(dense1)
    
    model = Model(inputs=[input_1, input_2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pairs, pair_labels, test_size=0.2, random_state=42)

X_train_1 = np.array([x[0] for x in X_train])
X_train_2 = np.array([x[1] for x in X_train])
X_test_1 = np.array([x[0] for x in X_test])
X_test_2 = np.array([x[1] for x in X_test])

model = create_rnn_model()

history = model.fit([X_train_1, X_train_2], y_train,
                    validation_data=([X_test_1, X_test_2], y_test),
                    epochs=10,  # Adjust epochs and batch_size as necessary
                    batch_size=32)

model.save('rnn_model.h5')

test_loss, test_accuracy = model.evaluate([X_test_1, X_test_2], y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")