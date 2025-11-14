import streamlit as st
import numpy as np
import pickle
import re
import contractions
import spacy
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .positive-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #c3e6cb;
        border: 2px solid #28a745;
        margin: 10px 0;
    }
    .negative-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f5c6cb;
        border: 2px solid #dc3545;
        margin: 10px 0;
    }
    .confidence-text {
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# Load models and configuration
@st.cache_resource
def load_models():
    """Load all required models and configurations"""
    try:
        # Load LSTM model
        model = load_model('sentiment_lstm_word2vec_model.h5')

        # Load Word2Vec model
        word2vec_model = Word2Vec.load('word2vec_model.bin')

        # Load configuration
        with open('lstm_word2vec_model_config.pkl', 'rb') as f:
            model_config = pickle.load(f)

        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")

        return model, word2vec_model, model_config, nlp
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure all model files are in the same directory as this script.")
        return None, None, None, None


# Preprocessing function
def preprocess_review(review, nlp):
    """Apply the same cleaning steps used during training"""
    # Remove emails
    review = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', review)
    # Remove HTML tags
    review = re.sub(r"<.*?>", "", review)
    # Lowercase
    review = review.lower()
    # Fix contractions
    review = contractions.fix(review)
    # Lemmatize
    review = " ".join(token.lemma_ for token in nlp(review))
    return review


# Prediction function
def predict_sentiment(review_text, model, vocab, max_length, nlp):
    """
    Predict sentiment for a single review

    Args:
        review_text: String containing the raw review
        model: Loaded LSTM model
        vocab: Vocabulary mapping
        max_length: Maximum sequence length
        nlp: spaCy model

    Returns:
        sentiment: 'Positive' or 'Negative'
        confidence: Probability score
        cleaned_text: Preprocessed review
    """
    # Preprocess the review
    cleaned_text = preprocess_review(review_text, nlp)

    # Tokenize
    tokens = cleaned_text.split()

    # Convert to sequence
    sequence = [vocab.get(word, 0) for word in tokens]

    # Pad sequence
    padded = pad_sequences([sequence], maxlen=max_length, padding='post')

    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]

    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return sentiment, confidence, cleaned_text


# Main app
def main():
    # Header
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("### Analyze the sentiment of movie reviews using Deep Learning")
    st.markdown("---")

    # Load models
    with st.spinner("Loading models..."):
        model, word2vec_model, model_config, nlp = load_models()

    if model is None:
        st.stop()

    vocab = model_config['vocab']
    max_length = model_config['max_length']

    st.success("✅ Models loaded successfully!")

    # Example buttons
    st.markdown("**Try an example:**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Positive Example", use_container_width=True):
            st.session_state.review_text = "This movie was absolutely fantastic! The cinematography was breathtaking and the performances were outstanding. I highly recommend it to everyone."

    with col2:
        if st.button("👎 Negative Example", use_container_width=True):
            st.session_state.review_text = "What a waste of time. The plot was confusing, the acting was terrible, and I couldn't wait for it to end. Definitely not worth watching."

    # Input section
    st.markdown("### Enter a Movie Review")
    review_input = st.text_area(
        label="Type or paste your review here:",
        placeholder="e.g., This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        height=150,
        label_visibility="collapsed",
        value=st.session_state.get('review_text', '')
    )

    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

    # Prediction
    if analyze_button:
        if review_input.strip() == "":
            st.warning("⚠️ Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing..."):
                sentiment, confidence, cleaned_text = predict_sentiment(
                    review_input, model, vocab, max_length, nlp
                )

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            if sentiment == 'Positive':
                st.markdown(f"""
                    <div class="positive-box">
                        <h2 style="color: #28a745; margin: 0;">😊 Positive Review</h2>
                        <p class="confidence-text">Confidence: {confidence * 100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="negative-box">
                        <h2 style="color: #dc3545; margin: 0;">😞 Negative Review</h2>
                        <p class="confidence-text">Confidence: {confidence * 100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

            # Progress bar for confidence
            st.progress(float(confidence))

            # Show preprocessed text in expander
            with st.expander("🔧 View Preprocessed Text"):
                st.text(cleaned_text)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Powered by LSTM + Word2Vec | Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()