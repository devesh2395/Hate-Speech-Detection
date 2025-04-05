# 🛡️ Hate Speech Detection GUI

A simple GUI-based application using Python and machine learning to classify tweets or text input into:
- Hate Speech Detected
- Offensive Language Detected
- No Hate Speech/Offensive Language Detected

This app uses a **Decision Tree Classifier** trained on a Twitter dataset, with text preprocessing and feature extraction using `CountVectorizer`.

---

## 📦 Features

- Clean and minimal GUI using `tkinter`
- Real-time text classification
- Text preprocessing (stopwords removal, punctuation cleaning, URL and number stripping)
- Machine learning pipeline with Scikit-learn

---

## 🛠️ Installation

### ✅ Requirements
Install the following libraries using `pip`:

```bash
pip install numpy pandas scikit-learn nltk
```

### 📁 Dataset
Ensure you have a CSV file named `twitter_data.csv` in the same directory.  
The CSV should contain at least two columns:  
- `tweet` (text content)
- `class` (integer label: 0, 1, or 2)

---

## 🧠 How It Works

### Label Mapping:
| Class | Description |
|-------|-------------|
| 0     | Hate Speech Detected |
| 1     | Offensive Language Detected |
| 2     | No Hate Speech/Offensive Language Detected |

### Preprocessing Includes:
- Lowercasing
- Removing punctuation and HTML tags
- Removing stopwords using `nltk`
- Stripping URLs and numbers

---

## 🚀 Usage

1. Run the script:

```bash
python hate_speech_gui.py
```

2. Enter your text or tweet in the input field.

3. Click **"Analyze"**.

4. A popup will show the predicted category.

---

## 🧪 Model Details

- **Vectorizer**: `CountVectorizer`
- **Classifier**: `DecisionTreeClassifier` from Scikit-learn
- **Training/Test Split**: 67% train, 33% test

---

## 📸 GUI Preview

*(Insert a screenshot of the app here if available)*

---

## 📚 Notes

- This project uses a basic ML model for demonstration and educational purposes. It can be improved with:
  - Better preprocessing (lemmatization, stemming)
  - Advanced models (e.g., Random Forest, SVM, or Deep Learning)
  - More robust datasets

---

## 👨‍💻 Author

Developed by **devesh2395**  
Feel free to fork, improve, and contribute!
