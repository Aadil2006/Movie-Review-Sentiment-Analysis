import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# safely downloading stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stops = set(stopwords.words('english'))

def text_cleaner(review):
    review = re.sub(r'<.*?>', ' ', review)
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    
    words = review.split()
    cleaned_words = [w for w in words if w not in stops]
    return ' '.join(cleaned_words)

try:
    with open('sentiment_model.pkl', 'rb') as f1:
        my_model = pickle.load(f1)
    
    with open('vectorizer.pkl', 'rb') as f2:
        my_vectorizer = pickle.load(f2)
except Exception as e:
    st.error(f"Error loading model files! {e}")

# ---- Web App UI ----
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Naviotech Internship Project | Developed by Aadil")

user_review = st.text_area("Enter your movie review here:", "")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        clean_rev = text_cleaner(user_review)
        vec_rev = my_vectorizer.transform([clean_rev])
        
        # Predicting Probability
        probabilities = my_model.predict_proba(vec_rev)[0]
        
        prob_negative = probabilities[0]
        prob_positive = probabilities[1]
        
        # UPGRADED LOGIC: Made the Neutral gap much wider (30% to 70%)
        # Ab AI ko Pure Positive bolne ke liye kam se kam 70% sure hona padega
        if 0.30 <= prob_positive <= 0.70:
            st.info("Result: This is a NEUTRAL / MIXED review. 😐")
            st.caption(f"AI Confidence Breakdown: {prob_positive*100:.1f}% Positive | {prob_negative*100:.1f}% Negative")
            
        elif prob_positive > 0.70:
            st.success("Result: This is a POSITIVE review. 😃")
            st.caption(f"AI Confidence: {prob_positive*100:.1f}% Sure")
            
        else:
            st.error("Result: This is a NEGATIVE review. 😞")
            st.caption(f"AI Confidence: {prob_negative*100:.1f}% Sure")