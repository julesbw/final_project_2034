import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load model and vectorizer
@st.cache_resource
def load_components():

    model = load_model("model_transformer_2.h5", compile=False)
    vectorizer = load_model("vectorizer.keras", compile=False)
    return model, vectorizer

# Predict function
def predict_text(text, model, vectorizer, threshold=0.5):
    input_tensor = vectorizer([text])
    prob = model.predict(input_tensor)[0][0]
    label = "Real" if prob >= threshold else "Fake"
    return label, float(prob)

# Streamlit interface
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Clasifica noticias como reales o falsas usando un modelo Transformer entrenado en Keras.")

model, vectorizer = load_components()
user_input = st.text_area("Escribe o pega una noticia:", height=200)

if st.button("Predecir"):
    if user_input.strip() == "":
        st.warning("Por favor, escribe una noticia primero.")
    else:
        label, prob = predict_text(user_input, model, vectorizer)
        st.markdown(f"### Resultado: **{label}**")
        st.markdown(f"Probabilidad: **{prob:.2%}**")