
import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
import spacy

# --- NLTK and SpaCy Downloads (Conditional for Streamlit deployment) ---
@st.cache_resource
def download_nltk_data_app():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    return True

@st.cache_resource
def load_spacy_model_app():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.write("Downloading SpaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

# Perform downloads once at app startup
download_nltk_data_app()
nlp_spacy_app = load_spacy_model_app()
stop_words_app = set(stopwords.words("english"))

# --- Preprocessing Functions ---
def clean_text_app(text):
    text = str(text).lower() # Ensure text is string and convert to lowercase
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def preprocess_app(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words_app]
    return " ".join(tokens)

# --- Load Models ---
@st.cache_resource
def load_lstm_model_app():
    return load_model("lstm_model.h5")

@st.cache_resource
def load_tokenizer_app():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model_app = load_lstm_model_app()
tokenizer_app = load_tokenizer_app()

# --- Hugging Face Pipeline ---
@st.cache_resource
def load_hf_pipeline_app():
    return pipeline("sentiment-analysis")

sentiment_pipeline_hf_app = load_hf_pipeline_app()

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")

st.title("Comprehensive Sentiment Analysis and NLP Dashboard")
st.markdown('''
Welcome to this interactive dashboard! This application demonstrates various Natural Language Processing (NLP) techniques,
including text preprocessing, sentiment analysis using a custom LSTM model and Hugging Face Transformers,
and Named Entity Recognition (NER) with SpaCy.
''')

# Section 1: Text Preprocessing
st.header("1. Text Preprocessing")
st.markdown('''
This section illustrates how raw text is cleaned and preprocessed for NLP tasks.
We apply two main steps:
- **Cleaning**: Converts text to lowercase, removes URLs, and removes special characters.
- **Preprocessing**: Tokenizes the text and removes common English stopwords.
''')

preprocessing_input_app = st.text_area("Enter text for preprocessing:", "This is an example sentence with a URL: https://example.com and some !@#$ special characters.")
if st.button("Show Preprocessing Steps"):
    st.write(f"**Original Text:** {preprocessing_input_app}")
    cleaned_output_app = clean_text_app(preprocessing_input_app)
    st.write(f"**Cleaned Text:** {cleaned_output_app}")
    processed_output_app = preprocess_app(cleaned_output_app)
    st.write(f"**Processed Text (stopwords removed):** {processed_output_app}")

st.markdown("---")

# Section 2: Custom LSTM Sentiment Prediction
st.header("2. Custom LSTM Model Sentiment Prediction")
st.markdown('''
This section uses a Long Short-Term Memory (LSTM) neural network model, trained on custom data,
to predict the sentiment of your input text.
''')

lstm_input_app = st.text_area("Enter text for LSTM sentiment prediction:", "I love this product, it's absolutely fantastic!")

if st.button("Predict with LSTM"):
    # Clean and preprocess the input
    cleaned_text_lstm_app = clean_text_app(lstm_input_app)
    processed_text_lstm_app = preprocess_app(cleaned_text_lstm_app)

    # Convert to sequence and pad
    seq_app = tokenizer_app.texts_to_sequences([processed_text_lstm_app])
    # Ensure maxlen matches the maxlen used during model training
    padded_app = pad_sequences(seq_app, maxlen=100) 

    # Predict
    prediction_app = model_app.predict(padded_app)[0][0]

    st.write(f"**Processed Text:** {processed_text_lstm_app}")
    if prediction_app > 0.5:
        st.success(f"**Prediction:** Positive Sentiment 😊 (Score: {prediction_app:.4f})")
    else:
        st.error(f"**Prediction:** Negative Sentiment 😔 (Score: {prediction_app:.4f})")

st.markdown("---")

# Section 3: Hugging Face Transformers Sentiment Analysis
st.header("3. Hugging Face Transformers Sentiment Analysis")
st.markdown('''
This section utilizes a pre-trained sentiment analysis model from the Hugging Face Transformers library
to determine the sentiment of your text.
''')

hf_input_app = st.text_area("Enter text for Hugging Face sentiment analysis:", "I am very happy with the service today.")
if st.button("Predict with Hugging Face"):
    hf_result_app = sentiment_pipeline_hf_app(hf_input_app)
    st.write(f"**Prediction:** {hf_result_app[0]['label']} (Score: {hf_result_app[0]['score']:.4f})")

st.markdown("---")

# Section 4: SpaCy Named Entity Recognition (NER)
st.header("4. SpaCy Named Entity Recognition (NER)")
st.markdown('''
This section demonstrates Named Entity Recognition (NER) using SpaCy, which identifies and classifies
named entities in text into predefined categories like person, organization, locations, etc.
''')

ner_input_app = st.text_area("Enter text for NER with SpaCy:", "Apple Inc. was founded by Steve Jobs in California.")
if st.button("Analyze with SpaCy NER"):
    doc_app = nlp_spacy_app(ner_input_app)
    if doc_app.ents:
        st.write("Identified Entities:")
        for ent in doc_app.ents:
            st.markdown(f"- **{ent.text}** (_{ent.label_}_)")
    else:
        st.write("No named entities found.")
