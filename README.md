# Credit Card Approval Prediction System

This project is a complete machine learning web application for predicting whether a credit card application is likely to be approved or rejected.

Banks receive a large number of applications every day. Reviewing every profile manually can be slow, repetitive, and error-prone. This application uses historical applicant patterns to make a fast first-level screening decision. It is built with Python, Flask, scikit-learn, XGBoost, Pandas, Matplotlib, and Seaborn.

## Features

- Credit card approval prediction
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Flask Web Application
- Model Evaluation
- IBM Watson deployment support

## Project Structure

Credit Card Approval Prediction/
│
├── app.py
├── credit_card_approval/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── prediction.py
│   ├── schemas.py
│   └── visualization.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── Project Documentation.docx
│
├── models/
│   ├── credit_card_approval_model.joblib
│   └── model_metrics.json
│
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── static/
├── templates/
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-watson.txt

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python scripts/train_model.py
```

Run the Flask application:

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

## Dataset

The project uses:

- application_record.csv
- credit_record.csv

## Future Enhancements

- IBM Watson Machine Learning Deployment
- Explainable AI
- Real-time Prediction API
- Cloud Deployment

## Author

**JADA BHARGAVI**