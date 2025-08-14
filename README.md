# NLP-Sentiment-Analysis
Developed a Natural Language Processing (NLP) model to classify text sentiment as positive, negative, or neutral. Implemented data preprocessing (tokenization, stopword removal) and trained a machine learning model, achieving high accuracy in sentiment prediction on a labeled dataset.
# 🗣 NLP Sentiment Analysis (IMDB Reviews)

This project performs **sentiment analysis** on movie reviews from the IMDB dataset, classifying them as **positive** or **negative**. The model uses **TF-IDF vectorization** and a **Logistic Regression** classifier to achieve high accuracy.


## 📌 Project Overview
- **Dataset:** IMDB Movie Reviews (50,000 labeled reviews)
- **Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Model:** Logistic Regression (fast and effective for text classification)
- **Goal:** Predict sentiment from raw text reviews
- **Accuracy Achieved:** ~90%


## 🚀 Features
- Loads and preprocesses real-world text data
- Uses TF-IDF for high-quality text representation
- Trains a Logistic Regression model
- Evaluates using accuracy, classification report, and confusion matrix
- Visualizes results with Seaborn heatmap


## 📊 Results
| Metric       | Score |
|--------------|-------|
| Accuracy     | ~90%  |
| Precision    | High  |
| Recall       | High  |

## 🛠 Tools & Libraries
- **Python**
- **TensorFlow/Keras** (for dataset loading)
- **Scikit-learn**
- **Matplotlib** & **Seaborn**


## 📜 How to Run
1. Open the notebook in [Google Colab]([https://colab.research.google.com/](https://colab.research.google.com/drive/16rJEzTFTWE_pO99eSHiCIY-4kyRJLKHD#scrollTo=SBBQwDETbyUV)).
2. Upload the `.ipynb` file.
3. Run all cells to train and evaluate the model.


## 📈 Future Improvements
- Use **LSTM** or **BERT** for deeper contextual understanding
- Expand to **multi-class sentiment** (e.g., neutral)
- Deploy as a **web app** using Flask or Streamlit

