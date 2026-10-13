# ============================================================
# MUSICAL INSTRUMENT REVIEW SENTIMENT ANALYSIS
# STREAMLIT INTERFACE
# ============================================================

import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Musical Instrument Sentiment Analyzer",
    page_icon="🎵",
    layout="centered"
)


# ============================================================
# 2. DOWNLOAD NLTK RESOURCES
# ============================================================

nltk.download("stopwords")
nltk.download("wordnet")


# ============================================================
# 3. LOAD TRAINED MODEL AND TF-IDF
# ============================================================

@st.cache_resource
def load_model():

    svm_model = joblib.load(
        "svm_sentiment_model.pkl"
    )

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return svm_model, tfidf


svm_model, tfidf = load_model()


# ============================================================
# 4. NLP PREPROCESSING
# ============================================================

stop_words = set(
    stopwords.words("english")
)


# Keep negation words because they affect sentiment

negation_words = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "cannot",
    "couldn't",
    "wouldn't",
    "shouldn't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "wasn't",
    "weren't",
    "can't",
    "won't"
}


# Remove negation words from stopwords

stop_words = stop_words - negation_words


lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    # Convert back to sentence
    return " ".join(words)


# ============================================================
# 5. HEADER
# ============================================================

st.title("🎵 Musical Instrument Review Sentiment Analyzer")

st.write(
    "Enter a musical instrument review and the trained "
    "SVM model will classify it as Positive, Negative, "
    "or Neutral."
)

st.divider()


# ============================================================
# 6. REVIEW INPUT
# ============================================================

review = st.text_area(
    "Enter your review:",
    placeholder=(
        "Example: The guitar has amazing sound quality "
        "and I absolutely love it!"
    ),
    height=150
)


# ============================================================
# 7. ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if review.strip() == "":

        st.warning(
            "Please enter a review first."
        )

    else:

        # ----------------------------------------------------
        # PREPROCESS REVIEW
        # ----------------------------------------------------

        cleaned_review = preprocess_text(
            review
        )


        # ----------------------------------------------------
        # TF-IDF TRANSFORMATION
        # ----------------------------------------------------

        review_tfidf = tfidf.transform(
            [cleaned_review]
        )


        # ----------------------------------------------------
        # SVM PREDICTION
        # ----------------------------------------------------

        prediction = svm_model.predict(
            review_tfidf
        )[0]


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Sentiment Analysis Result"
        )


        if prediction == "Positive":

            st.success(
                "😊 Positive Review"
            )

        elif prediction == "Negative":

            st.error(
                "😞 Negative Review"
            )

        else:

            st.info(
                "😐 Neutral Review"
            )


        # ----------------------------------------------------
        # SHOW DETAILS
        # ----------------------------------------------------

        st.write("### Your Review")

        st.write(review)


        st.write("### Processed Review")

        st.write(cleaned_review)


        st.write("### Predicted Sentiment")

        st.write(
            f"**{prediction}**"
        )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        This application uses Natural Language Processing
        and Machine Learning to classify musical instrument
        reviews.
        """
    )

    st.write("### Technologies Used")

    st.write(
        """
        • NLP  
        • Text Preprocessing  
        • Lemmatization  
        • TF-IDF  
        • SVM  
        • Streamlit  
        """
    )

    st.write("### Model")

    st.write(
        "Linear Support Vector Machine (LinearSVC)"
    )

    st.write("### Classes")

    st.write(
        """
        😊 Positive  
        😐 Neutral  
        😞 Negative
        """
    )