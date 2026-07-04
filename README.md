# Tweet Emotion Prediction 🎭

An LSTM-based deep learning model that predicts the emotion behind a piece of text, 
with a live Gradio demo.

## Features
- Text preprocessing & tokenization pipeline (Keras Tokenizer, 10k vocab)
- LSTM neural network classifying text into 13 emotions
- Pre-trained model included (`emotion_model.h5`) — no retraining needed
- Interactive web UI via Gradio

## Dataset
[tweet_emotions.csv](tweet_emotions.csv) — 40,000 labeled tweets across 13 emotions 
(neutral, worry, happiness, sadness, love, surprise, fun, relief, hate, empty, 
enthusiasm, boredom, anger). Source: [Kaggle - Emotion Dataset for NLP](https://www.kaggle.com/) *(update with actual source link)*

## Model
- Embedding(10000, 128) → LSTM(128) → Dense(128, ReLU) → Dense(13, Softmax)
- Trained for 10 epochs, batch size 32

## Installation
​```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
​```

## Usage
​```bash
python emotion_prediction.py
​```
This launches a Gradio app in your browser where you can type text and get a predicted emotion.

## Requirements
​```
tensorflow
pandas
numpy
scikit-learn
gradio
​```

## Limitations
- Class imbalance (e.g., "anger" has only 110 samples vs "neutral"'s 8,638) can 
  bias predictions toward majority classes
- Tokenizer is refit on data at runtime rather than saved/loaded separately — 
  for production use, the tokenizer and label encoder should be pickled alongside the model

## Future Improvements
- Save/load tokenizer & label encoder (currently rebuilt on every run)
- Address class imbalance (class weights, oversampling)
- Try Bidirectional LSTM / GRU / transformer-based models (e.g. BERT)
- Add evaluation metrics (F1, confusion matrix) to the repo

## License
MIT