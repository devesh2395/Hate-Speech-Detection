import pandas as pd
import numpy as np
import re
import string
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("twitter_data.csv")

# Correct class mapping
df['labels'] = df['class'].map({0: "Hate Speech Detected", 
                                1: "Offensive Language Detected", 
                                2: "No Hate Speech/Offensive Language detected"})

# Selecting only relevant columns
df = df[['tweet', 'labels']]

# Stopwords and punctuation removal function
stopword = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove new lines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    text = " ".join([word for word in text.split() if word not in stopword])  # Remove stopwords
    return text

df["tweet"] = df["tweet"].apply(clean)

# Prepare data for model training
x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Test cases
test_cases = ["I will hurt you", "I love this place", "You are stupid"]
for test in test_cases:
    transformed_test = cv.transform([test])
    print(f"Text: '{test}' → Prediction: {clf.predict(transformed_test)[0]}")
