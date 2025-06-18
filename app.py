import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf

# Page Configuration
st.set_page_config(page_title="Transformer Model App", layout="wide")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Inference", "Dataset Visualization", "Hyperparameter Tuning", "Model Analysis"],
        icons=["robot", "bar-chart", "sliders", "clipboard-data"],
        menu_icon="cast",
        default_index=0
    )



@st.cache_resource
def load_model_and_vectorizer():
    model = tf.keras.models.load_model("models/model_transformer_2.h5", compile=False)
    vectorizer = tf.keras.models.load_model("models/vectorizer_2.keras", compile=False)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
# Inference Page
if selected == "Inference":
    st.title("Inference Interface")
    st.write("Input a sentence to get the model's prediction and confidence.")

    user_input = st.text_area("Enter your text here:")

    if st.button("Classify") and user_input.strip() != "":
        # Vectorize input
        input_tensor = vectorizer(tf.constant([user_input]))
        
        # Predict
        prediction = model.predict(input_tensor)[0][0]
        label = "Real" if prediction > 0.5 else "Fake"
        confidence = float(prediction) if label == "Real" else 1 - float(prediction)

        # Display Results
        st.success(f"**Prediction**: {label}")
        st.info(f"**Confidence**: {confidence:.2%}")

# Dataset Visualization Page
elif selected == "Dataset Visualization":
    st.title("Dataset Visualization")
    st.write("This section displays insights from the dataset.")

    # Token Length Histogram
    st.subheader("Token Length Histogram")
    st.image("images/token_length_hist.png", use_column_width=True)

    # Word Cloud
    st.subheader("Word Cloud")
    st.image("images/wordcloud.png", use_column_width=True)


# Hyperparameter Tuning Page
elif selected == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    st.write("This section shows results from hyperparameter optimization using Optuna.")

    st.subheader("Tuned Hyperparameters")
    st.markdown("""
    - **Embedding Dimension**: 128  
    - **Number of Attention Heads**: 2  
    - **Feedforward Dimension**: 128  
    - **Dropout Rate**: 0.298  
    - **Learning Rate**: 2.17e-5  
    """)

    st.subheader("Performance Over Trials")
    st.image("images/optuna_optimization_history.png")

    st.subheader("Hyperparameter Importance")
    st.image("images/optuna_param_importances.png")

# Model Analysis Page
elif selected == "Model Analysis":
    st.title("Model Analysis and Justification")
    st.write("This page explains model performance, dataset challenges, and detailed error analysis.")

    st.subheader("Classification Report")
    st.text("""
Classification Report:
              precision    recall  f1-score   support
    Fake          0.95      0.94      0.94      4669
    Real          0.94      0.94      0.94      4311

    accuracy                           0.94      8980
    macro avg      0.94      0.94      0.94      8980
    weighted avg   0.94      0.94      0.94      8980
    """)

    st.subheader("Confusion Matrix")
    st.image("images/confusion_matrix.png")

    st.subheader("Error Analysis")
    st.markdown("""
    - Most errors occur in borderline cases where the text is ambiguous or lacks strong signals of veracity.
    - False positives are sometimes due to sensational language used in real headlines.
    - False negatives include fake headlines that mimic journalistic tone or cite credible sources.
    - Model could improve with:
        - More annotated training data
        - Context-aware embeddings (e.g., adding surrounding paragraph)
        - Ensemble strategies combining rule-based filters with deep learning
    """)
