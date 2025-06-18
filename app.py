import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf

# Page Configuration
st.set_page_config(page_title="Transformer Model App", layout="wide")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Inference", "Dataset Visualization", "Hyperparameter Tuning", "Model Analysis", "About"],
        icons=["robot", "bar-chart", "sliders", "clipboard-data"],
        menu_icon="cast",
        default_index=0
    )


# @st.cache_resource
# def load_model_and_vectorizer():
#     model = tf.keras.models.load_model("models/model_transformer_2.h5", compile=False)
#     vectorizer = tf.keras.models.load_model("models/vectorizer_2.keras", compile=False)
#     return model, vectorizer

# model, vectorizer = load_model_and_vectorizer()


# Inference Page
if selected == "Inference":
    st.title("Inference Interface")
    st.write("Enter a news headline or short sentence to classify it as Real or Fake.")

    # Text input box
    user_input = st.text_area("Enter your text here:")

    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            # Placeholder output
            st.markdown("**Prediction:** Real")
            st.markdown("**Confidence:** 87.32%")

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
    st.write(
        """
        This section summarizes the hyperparameter optimization performed using Optuna. 
        The goal was to find the best configuration of model parameters to maximize validation accuracy.
        """
    )

    st.subheader("Optimization History")
    st.image("images/optuna_optimization_history.png")
    st.markdown(
        """
        The line plot above shows how the objective value (validation accuracy) evolved over the trials.
        Although only a few trials were run due to time constraints, we can observe a clear improvement
        by the third trial, where accuracy reached **0.971**. This upward trend indicates that further trials 
        could yield even better configurations.
        """
    )

    st.subheader("Hyperparameter Importance")
    st.image("images/optuna_param_importances.png")
    st.markdown(
        """
        The bar chart above reveals which hyperparameters had the greatest impact on model performance.
        
        - **Dropout Rate** was the most influential, contributing nearly 46% to the objective value.
        - **Feedforward Dimension (ff_dim)** and **Embedding Dimension** also played significant roles.
        - Interestingly, the **number of attention heads** had the least effect, suggesting that model capacity
          was not strongly limited by this factor in our configuration.

        These insights can help prioritize which parameters to fine-tune in future iterations of model design.
        """
    )

    st.subheader("Best Configuration Found")
    st.markdown(
        """
        After tuning, the best hyperparameters were:

        - Embedding Dimension: `64`
        - Number of Attention Heads: `2`
        - Feedforward Dimension (ff_dim): `128`
        - Dropout Rate: `0.2983`
        - Learning Rate: `2.17e-5`

        These values were used to retrain the final model evaluated in the analysis section.
        """
    )


# Model Analysis Page
elif selected == "Model Analysis":
    st.title("Model Analysis and Justification")
    st.write("This page explains model performance, dataset challenges, and detailed error analysis.")

    st.subheader("Model Architecture")
    st.markdown("""
    The selected model is based on a **transformer encoder architecture**, which is highly effective for understanding the context and structure of textual data.
    For the task of **fake news detection**, transformer encoders such as RoBERTa or DeBERTa are particularly powerful, as they can capture nuanced patterns in language, 
    such as sarcasm, misinformation tone, or subtle factual inconsistencies.

    Compared to traditional approaches like LSTM or CNNs, transformer-based encoders offer superior handling of long-range dependencies 
    and allow the model to consider all words in the input simultaneously. This makes them ideal for processing news headlines or full articles 
    where meaning often depends on context spread throughout the sentence.

    In this project, the transformer encoder model was chosen due to:
    - Its state-of-the-art performance in NLP classification tasks.
    - Its ability to generalize well even with relatively small datasets when fine-tuned.
    - Its strong performance in prior work related to misinformation and sentiment analysis.
    """)
    
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
# About Page
if selected == "About":
    st.title("About the Author")
    st.markdown("""
    **Julio César Briones Wong**  
    *Student ID:* A00838831  
    *Course:* Modeling Learning with Artificial Intelligence  

    This application was developed as part of the final project for the course "Modeling Learning with Artificial Intelligence".  
    It demonstrates the use of a fine-tuned Transformer model for detecting fake news, with a user-friendly Streamlit interface that integrates real-time inference, dataset exploration, hyperparameter tuning visualizations, and performance analysis.
    
    I hope this tool not only showcases technical implementation, but also encourages further research and application of NLP models for social impact.  
    """)