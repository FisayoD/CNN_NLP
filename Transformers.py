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


# Assuming 'documents' is a list of texts loaded from your file paths
labels = [1 if "oil" in doc.lower() else 0 for doc in documents]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized = tokenizer(documents, padding=True, truncation=True, return_tensors="tf")

input_ids_np = tokenized['input_ids'].numpy()
attention_mask_np = tokenized['attention_mask'].numpy()

# Then, use the numpy arrays with train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_ids_np, labels, test_size=0.2, random_state=42)
train_mask, test_mask = train_test_split(attention_mask_np, test_size=0.2, random_state=42)

# Now you can convert them back to TensorFlow tensors if needed for your model
X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
train_mask = tf.constant(train_mask)
test_mask = tf.constant(test_mask)

def create_transformer_model():
    # Load pre-trained BERT model
    bert = TFBertModel.from_pretrained('bert-base-uncased')

    # Inputs
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
    
    # Get the embeddings (last hidden states) from BERT
    bert_output = bert(input_ids)[0]

    # We take the representation of the [CLS] token to use for classification.
    cls_token = bert_output[:, 0, :]

    # Add a dense layer for classification
    out = Dense(1, activation='sigmoid')(cls_token)

    # Final model
    model = Model(inputs=input_ids, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Model creation and training
model = create_transformer_model()

# Show the model architecture
model.summary()

# Now, train the model with the correct tensors
history = model.fit(
    {'input_ids': X_train, 'attention_mask': train_mask},
    y_train,
    validation_data=({'input_ids': X_test, 'attention_mask': test_mask}, y_test),
    epochs=3,
    batch_size=8
)

# Save the model
model.save('transformer_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate({'input_ids': X_test, 'attention_mask': test_mask}, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")