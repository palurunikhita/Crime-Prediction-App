# Crime-Prediction-App

This is a web-based application that predicts the likelihood of a crime in LA based on input features like time, location, and victim demographics using trained machine learning models.

## Features

- Predicts crime likelihood levels (Very Low → Very High)
- Compares multiple ML models and selects the best
- Accuracy and computation time visualizations
- Zipped dataset handling to stay under GitHub’s file size limits
- Responsive UI with dark mode toggle

## Dataset

- The dataset is zipped as `CrimeDataset.csv.zip` to comply with GitHub’s file size limit.
- The app will auto-extract the file if `CrimeDataset.csv` is missing.

## Setup Instructions

```
git clone git@github.com:palurunikhita/Crime-Prediction-App.git
cd CrimePredictionApp
python3 -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open your browser and go to: `http://127.0.0.1:5000`

## Machine Learning Models Used

The app trains and evaluates the following models:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting (GB)  
- Decision Tree (DT)  
- XGBoost  

The best-performing model (based on highest accuracy and lowest computation time) is auto-selected and saved as `model.pkl`.

## Charts Rendered in App

- Accuracy Comparison  
- Computation Time Comparison

## Tech Stack

- **Frontend**: HTML, CSS (with dark mode)  
- **Backend**: Python + Flask  
- **ML**: Scikit-learn, XGBoost, Pandas, Numpy, Matplotlib, Seaborn  
- **Model Serialization**: `joblib`
