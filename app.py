import pandas as pd
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---------- Preprocessing Function ----------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letter characters
    text = text.lower()                     # Lowercase
    text = text.strip()                     # Remove leading/trailing whitespace
    return text

# ---------- Load Data ----------
# Example: replace this with your actual dataset path
df = pd.read_csv("spam.csv")  # Ensure this file has 'email' and 'label' columns

# Drop rows with missing email or label
df = df[df['email'].notna() & df['label'].notna()]

# Preprocess emails
df['email'] = df['email'].apply(preprocess_text)

# ---------- Feature & Label Split ----------
X = df['email']
y = df['label']

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Pipeline: TF-IDF + Naive Bayes ----------
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# ---------- Train Model ----------
model.fit(X_train, y_train)

# ---------- Evaluate Model ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------- Streamlit UI ----------
st.title("📧 Spam Email Classifier")

user_input = st.text_area("Enter your email content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        cleaned_input = preprocess_text(user_input)
        prediction = model.predict([cleaned_input])[0]
        st.success(f"Prediction: **{prediction.upper()}**")

st.sidebar.markdown("### Model Accuracy")
st.sidebar.write(f"{accuracy * 100:.2f}%")

# Optional: Classification report in debug mode
# st.text("Classification Report:")
# st.text(classification_report(y_test, y_pred))
