# 💳 Credit Card Fraud Detection

This machine learning project aims to identify fraudulent credit card transactions using classification techniques on a highly imbalanced dataset. The goal is to detect fraud with high recall while minimizing false positives.

---

## 📌 Project Summary

- **Domain**: Finance / Fraud Analytics  
- **Problem**: Imbalanced binary classification (fraud vs. normal)  
- **Model Used**: Random Forest with class balancing  
- **Techniques**: SMOTE, threshold tuning, feature importance

---

## 📊 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~285,000 transactions
- **Features**: 30 columns (PCA components: V1–V28, plus `Time`, `Amount`)
- **Target**: `Class` — `0` (legit), `1` (fraud)

> 📥 **To use this notebook**, download `creditcard.csv` from Kaggle and place it in the following folder:

<pre lang="markdown"> ## 📁 Project Structure ``` project-root/ ├── Anamoly detection.ipynb # Main Jupyter notebook ├── README.md # Project documentation └── data/ └── creditcard.csv # Dataset file (downloaded manually) ``` </pre>


---

## 🧠 Workflow

| Step                  | Description                                             |
|-----------------------|---------------------------------------------------------|
| 🧹 Data Cleaning       | Checked for nulls, distribution, and class imbalance    |
| ⚖️ SMOTE Oversampling | Balanced the training set using `SMOTE()`               |
| 🌲 Model Training     | Used `RandomForestClassifier` with `class_weight='balanced'` |
| 🎯 Threshold Tuning   | Adjusted the probability threshold to improve recall     |
| 📊 Evaluation         | Confusion Matrix, ROC-AUC, Classification Report         |
| ⭐ Feature Importance | Identified most influential features using `feature_importances_` |

---

## 📈 Model Evaluation

- **Recall (fraud class)**: High — focus was on catching as many frauds as possible  
- **ROC-AUC Score**: ~0.98  
- **Confusion Matrix**: Shows balance between FP and FN  
- **Most important features**: V14, V17, V10

---

## 📌 Key Libraries Used

- `pandas`, `numpy`
- `scikit-learn`
- `imbalanced-learn (SMOTE)`
- `matplotlib`, `seaborn`

---

## 🚀 Possible Improvements

- Add SHAP for model explainability
- Try ensemble models (XGBoost, LightGBM)
- Deploy as a Flask or Streamlit app
- Use hyperparameter tuning with GridSearchCV or RandomizedSearchCV

---

## 🤝 Acknowledgements

- Dataset by [ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 📎 Author

Ravishanker Gandigudi  
Master's in Computer Science, Stevens Institute of Technology  
[LinkedIn](https://www.linkedin.com/in/ravishanker-gandigudi/) | [GitHub](https://github.com/)

