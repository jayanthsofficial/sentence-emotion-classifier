import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import gradio as gr
import os

# Load and preprocess the dataset
data = pd.read_csv("tweet_emotions.csv")  # Ensure the correct filename

# Adjusting columns based on dataset structure
data = data[['content', 'sentiment']]  
data.columns = ["Text", "Emotions"]

# Convert text & labels to lists
texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
VOCAB_SIZE = 10000  # Limit vocabulary size for better efficiency
OOV_TOKEN = "<OOV>"  # Handle unknown words

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Encode the string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# One-hot encode the labels
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(set(labels)))

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
    padded_sequences, one_hot_labels, test_size=0.2, random_state=42
)

# Model file path
MODEL_FILE = "emotion_model.h5"

# Check if a trained model exists to avoid retraining
if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    print("Loaded saved model.")
else:
    # Define the model
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=max_length),
        LSTM(128, return_sequences=False),
        Dense(128, activation="relu"),
        Dense(len(one_hot_labels[0]), activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

    # Save the trained model
    model.save(MODEL_FILE)
    print("Model trained and saved.")

# Define prediction function for Gradio
def predict_emotion(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding="post")
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    return predicted_label[0]

# Gradio interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Enter a text here...", label="Input Text"),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="Emotion Prediction from Text",
    description="Enter a sentence to predict the associated emotion."
)

# Launch the interface
interface.launch()
