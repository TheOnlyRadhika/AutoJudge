# 🎯 AutoJudge – Problem Difficulty Predictor

AutoJudge is an end-to-end machine learning application that predicts the **difficulty level of programming problems** based on their textual content.  
The system analyzes problem titles and descriptions using NLP techniques and outputs a difficulty classification through an interactive **Streamlit web app**.

This project demonstrates a complete ML workflow — from preprocessing and feature engineering to model selection and deployment.

---

## 📌 Problem Statement

Online coding platforms host thousands of problems with varying difficulty levels.  
Manual difficulty tagging is subjective, inconsistent, and time-consuming.

**Goal:**  
To automatically predict the **difficulty level** of a problem using its **title and description**, ensuring consistency and scalability.

---

## 🧠 Approach Overview

The project follows a structured machine learning pipeline:

1. **Text preprocessing**
2. **Feature extraction (TF-IDF + engineered features)**
3. **Model training & evaluation**
4. **Best model selection**
5. **Deployment using Streamlit**

---

## 🧾 Input Features

The model takes the following inputs:

- **Problem Title** (text)
- **Problem Description** (text)

### Engineered Features
- TF-IDF vectors from title and description
- Text length and word-count based numerical features
- Combined sparse and dense features using `hstack`

---

## 🛠️ Text Preprocessing

Text data is cleaned and normalized using:
- Lowercasing
- Regex-based cleaning
- Whitespace normalization
- Removal of unwanted characters

Preprocessing is **identical during training and inference** to avoid data leakage.

---

## ⚙️ Feature Engineering

- TF-IDF vectorization for:
  - Problem titles
  - Problem descriptions
- Numeric features:
  - Character count
  - Word count
- Final feature matrix created by combining:
  - Sparse TF-IDF features
  - Dense numerical features

---

## 🤖 Models Trained

Multiple models were trained and evaluated, including:

- Logistic Regression (baseline)
- Tree-based models
- Gradient Boosting–based models

Each model was evaluated using appropriate performance metrics, and the **best-performing model** was selected for deployment.

---

## 🏆 Final Model

- Best model selected based on evaluation performance
- Model and vectorizers serialized using **pickle**
- Loaded safely during inference in the Streamlit app

---

## 🌐 Web Application (Streamlit)

The project includes an interactive **Streamlit web app** that:

- Accepts user input (title + description)
- Applies the same preprocessing and feature extraction
- Loads the trained model and vectorizers
- Outputs the predicted **difficulty level**

---

## 🗂️ Project Structure

```text
AutoJudge/
│
├── app.py                  # Streamlit application
├── model/
│   ├── trained_model.pkl   # Final trained model
│   ├── vectorizer.pkl      # TF-IDF vectorizers
│
├── notebooks/
│   ├── model_training.ipynb
│   ├── feature_engineering.ipynb
│
├── requirements.txt
├── README.md
