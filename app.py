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
    st.write("This section displays various insights from the dataset.")

    # Token Length Histogram
    st.subheader("Token Length Histogram")
    st.image("images/token_length_hist.png")
    st.markdown("""
    **Insight:** The histogram of token lengths indicates that most input texts have between 20 and 60 tokens. This confirms that the preprocessing step (e.g., truncation or padding) should consider a maximum length within this range. Longer texts may lose information if overly truncated, while shorter ones could lead to inefficient resource usage if padded excessively.
    """)

    # Word Cloud
    st.subheader("Word Cloud")
    st.image("images/wordcloud.png")
    st.markdown("""
    **Insight:** The most frequent words include political figures ("Trump", "Obama", "Clinton") and terms like "video", "tweet", and "attack". This suggests that the dataset is highly political and may reflect real-world media trends, potentially introducing bias. Prevalent named entities could also mislead a model into using entity presence rather than semantic context.
    """)

# Hyperparameter Tuning Page
elif selected == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    st.write("Below are the results from hyperparameter optimization.")

    st.subheader("Tuning Summary")
    st.markdown("""
    - **Tuned Parameters**: Embedding Dimension, Number of Attention Heads, Feed-forward Dimension, Dropout Rate, Learning Rate.
    - **Best Configuration**: Embedding Dim = 64, Heads = 2, FF Dim = 128, Dropout = 0.29, LR = 2.17e-5
    """)

    st.subheader("Performance over Trials")
    st.image("images/optuna_trials.png")
    st.markdown("""
    **Insight:** The performance curve shows that significant improvement occurred in the first few trials, with marginal gains thereafter. This indicates that early exploration yields the most impactful configurations, while later refinements provide diminishing returns.
    """)

    st.subheader("Best Parameters Visualized")
    st.image("images/optuna_best.png")
    st.markdown("""
    **Insight:** The best-performing configuration balanced model complexity with generalization by avoiding overly deep or wide layers. The low learning rate also prevented overshooting during optimization, contributing to more stable convergence.
    """)

# Model Analysis Page
elif selected == "Model Analysis":
    st.title("Model Analysis and Justification")
    st.write("This page explains model performance, dataset challenges, and detailed error analysis.")

    st.subheader("Classification Report")
    st.text("""
    Classification Report:

        precision    recall  f1-score   support

        Fake       0.95      0.94      0.94      4669
        Real       0.94      0.94      0.94      4311

        accuracy                         0.94      8980
        macro avg    0.94      0.94      0.94      8980
        weighted avg 0.94      0.94      0.94      8980
    """)
    st.markdown("""
    **Insight:** With an F1-score of 0.94 on both classes, the model demonstrates strong performance and balanced predictions. The macro and weighted averages confirm consistency across fake and real labels, suggesting no class imbalance or systemic bias in performance.
    """)

    st.subheader("Confusion Matrix")
    st.image("images/confusion_matrix.png")
    st.markdown("""
    **Insight:** The confusion matrix reveals relatively few false positives and false negatives. The classifier correctly identifies the majority of both classes, and the low off-diagonal values imply minimal confusion between categories.
    """)

    st.subheader("Error Analysis")
    st.markdown("""
    - **False Positives** may occur with satirical headlines or sensationalist language resembling misinformation.
    - **False Negatives** might be tied to subtle disinformation where tone and facts are hard to distinguish without external knowledge.
    - **Improvement Suggestions:**
        - Introduce external fact-checking features.
        - Fine-tune with more balanced real/fake subsets or adversarial samples.
        - Use transformer ensembles to aggregate diverse textual signals.
    """)
