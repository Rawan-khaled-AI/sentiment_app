import streamlit as st
import pickle

# Load vectorizer
with open(r'C:/Users/RAWAN/OneDrive/Desktop/pipline_internship/task_2/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load only Logistic Regression model
with open(r'C:/Users/RAWAN/OneDrive/Desktop/pipline_internship/task_2/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Correct label map (based on your training)
label_map = {
    0: 'Negative',
    1: 'Positive'
}

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.markdown("Type a sentence below and discover its **sentiment** using AI!")

# User input
user_input = st.text_area("💬 Enter your sentence:")

# Predict
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        sentiment = label_map.get(prediction, "Unknown")

        # Result with emoji and color
        if sentiment == "Positive":
            st.success(f"😊 Sentiment: **{sentiment}**")
        elif sentiment == "Negative":
            st.error(f"😞 Sentiment: **{sentiment}**")
        else:
            st.info(f"😐 Sentiment: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter a sentence to analyze.")

