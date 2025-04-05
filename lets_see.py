#Install Libs- pip install numpy pandas scikit-learn nltk

import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import string
import nltk
from nltk.corpus import stopwords

# Load necessary resources
nltk.download('stopwords')
stopword = set(stopwords.words("english"))

# Load and preprocess the dataset
df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({0:"Hate Speech Detected", 1:"Offensive Language Detected", 2:"No Hate Speech/Offensive Language detected"})
df = df[['tweet','labels']]

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

df["tweet"] = df["tweet"].apply(clean)

# Prepare data for training
x = np.array(df["tweet"])
y = np.array(df["labels"])
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

def predict():
    user_input = entry.get()
    cleaned_text = clean(user_input)
    vectorized_text = cv.transform([cleaned_text]).toarray()
    result = clf.predict(vectorized_text)[0]
    messagebox.showinfo("Prediction", f"Result: {result}")

# Create UI
root = tk.Tk()
root.title("Hate Speech Detector")
root.geometry("400x300")

label = tk.Label(root, text="Enter text:", font=("Arial", 12))
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

button = tk.Button(root, text="Analyze", command=predict)
button.pack(pady=20)

root.mainloop()
