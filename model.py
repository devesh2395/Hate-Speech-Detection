import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
stopword = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    return " ".join(text)

# Load and preprocess dataset
df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({
    0: "Hate Speech Detected",
    1: "Offensive Language Detected",
    2: "No Hate Speech/Offensive Language detected"
})
df = df[['tweet', 'labels']]
df['tweet'] = df['tweet'].apply(clean)

# Train model
x = df['tweet']
y = df['labels']
cv = CountVectorizer()
x = cv.fit_transform(x)
model = DecisionTreeClassifier()
model.fit(x, y)

# Prediction function
def predict_text(text: str):
    cleaned = clean(text)
    vector = cv.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]  # Use 'model' instead of 'clf'
    return prediction