import streamlit as st
import torch
import pickle
import json
import torch.nn as nn
import os
from torch.nn.utils.rnn import pack_padded_sequence
import re

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Compare **RNN** and **LSTM** sentiment models")

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cpu")

# =====================================================
# PARAMETERS
# =====================================================
BASE    = os.path.dirname(os.path.abspath(__file__))
maxlength = 595

# =====================================================
# TOKENIZER
# =====================================================
@st.cache_resource
def load_tokenizer():
    with open(os.path.join(BASE, "word_to_idx.pkl"), "rb") as f:
        data = pickle.load(f)
        print(data.keys())
    return data

wordtoidx = load_tokenizer()

# =====================================================
# SIMPLE TOKENIZER (same as training)
# =====================================================
def simple_tokenize(text):
    text = text.lower() #convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text) #remove all characters except alphanumeric and spaces
    return text.split()

# =====================================================
# TEXT ENCODING
# =====================================================
# Convert dataset to arrays for PyTorch
def encode_text(text, word_to_idx, max_len):
    tokens = simple_tokenize(text)
    # Convert to indices
    ids = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    length = min(len(ids), max_len)
    # Pad or truncate
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [word_to_idx['<PAD>']] * (max_len - len(ids))
    return ids,length

# =====================================================
# MODEL DEFINITIONS
# =====================================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size=1,
                 num_layers=1, bidirectional=False, dropout=0.3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, output_size)  # logits

    def forward(self, x, lengths):
        e = self.embedding(x)  # (B, T, E)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.rnn(packed)
        if self.rnn.bidirectional:
            h_fwd = h_n[-2]  # (B, H)
            h_bwd = h_n[-1]  # (B, H)
            last_h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        else:
            last_h = h_n[-1]  # (B, H)
        logits = self.fc(self.dropout(last_h)).squeeze(1)  # (B,)
        return logits


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size=1,
                 num_layers=1, bidirectional=False, dropout=0.3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, output_size)  # logits

    def forward(self, x, lengths):
        e = self.embedding(x)  # (B, T, E)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)  # LSTM returns (h_n, c_n)
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]  # (B, H)
            h_bwd = h_n[-1]  # (B, H)
            last_h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        else:
            last_h = h_n[-1]  # (B, H)
        logits = self.fc(self.dropout(last_h)).squeeze(1)  # (B,)
        return logits

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    lstm  = torch.load(os.path.join(BASE, "LSTM", "best_lstm_model_object.pt"), map_location="cpu", weights_only=False)
    rnn   = torch.load(os.path.join(BASE, "RNN",  "best_rnn_model_object.pt"), map_location="cpu", weights_only=False)
    lstm.eval()
    rnn.eval()
    return lstm, rnn

lstm_model, rnn_model = load_models()

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict(text, model):
    x, lengths = encode_text(text, wordtoidx, maxlength)
    prob = torch.sigmoid(model(torch.tensor([x], dtype=torch.long), torch.tensor([lengths], dtype=torch.long))).item()
    label = "Positive 🎉" if prob >= 0.5 else "Negative 💔"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence

# =====================================================
# USER INPUT
# =====================================================
review = st.text_area(
    "✍️ Enter a movie review:",
    height=150,
    placeholder="Example: This movie was amazing with great acting..."
)

# =====================================================
# PREDICT BUTTON
# =====================================================
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        lstm_label, lstm_prob = predict(review, lstm_model)
        rnn_label, rnn_prob = predict(review, rnn_model)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 LSTM Model")
            st.success(lstm_label)
            st.write(f"Confidence: {lstm_prob:.2%}")
            st.progress(float(lstm_prob))

        with col2:
            st.subheader("🔁 RNN Model")
            st.success(rnn_label)
            st.write(f"Confidence: {rnn_prob:.2%}")
            st.progress(float(rnn_prob))

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Built with PyTorch + Streamlit | IMDB Sentiment Classification")
