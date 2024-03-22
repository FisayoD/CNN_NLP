import tensorflow as tf
import os
import numpy as np
from keras.layers import Input, Dense
from transformers import BertTokenizer, TFBertModel, BertConfig
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split

HF_TOKEN = "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"

os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

output_txt_paths = [
    'articles_text/1intro.txt',
    'articles_text/2relation.txt',
    'articles_text/3count.txt',
    'articles_text/4probability.txt',
    'articles_text/libya.txt',
    'articles_text/Homework_3_Fisayo_Ojo.txt',
    'articles_text/An Investigation of the Pattern And Environmental Impact of Oil.txt',
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
        with open(path, 'r') as f:
            text = f.read()
            documents.append(text)
    except FileNotFoundError:
        print(f"File {path} not found. Skipping.")



labels = [1 if "oil" in doc.lower() else 0 for doc in documents]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized = tokenizer(documents, padding=True, truncation=True, max_length=512, return_tensors="tf")

input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_ids.numpy(), labels, test_size=0.2, random_state=42)
train_mask, test_mask = train_test_split(attention_mask.numpy(), test_size=0.2, random_state=42)

# Convert numpy arrays back to TensorFlow tensors
X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
train_mask = tf.constant(train_mask)
test_mask = tf.constant(test_mask)

MAX_SEQUENCE_LENGTH = 512 
def create_transformer_model():
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')   

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

   
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

  
    cls_token = bert_output[:, 0, :]

    output = Dense(1, activation='sigmoid')(cls_token)

    # Construct the final model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create, train, and evaluate the model as you have in your script
model = create_transformer_model()
model.summary()

# Training
history = model.fit([X_train, train_mask], y_train, validation_data=([X_test, test_mask], y_test), epochs=3, batch_size=8)

# Save the model
model.save('transformer_model.h5')

# Evaluate 
test_loss, test_accuracy = model.evaluate([X_test, test_mask], y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")