# Credit Card Approval Prediction System

This project is a complete machine learning web application for predicting whether a credit card application is likely to be approved or rejected.

Banks receive a large number of applications every day. Reviewing every profile manually can be slow, repetitive, and error-prone. This application uses historical applicant patterns to make a fast first-level screening decision. It is built with Python, Flask, scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn, and an optional IBM Watson Machine Learning deployment script.

The app is designed for learning, demos, and portfolio use. It should not be used as a real banking approval engine without proper compliance review, fairness testing, security hardening, and validation on verified financial data.

## What the Application Does

- Collects applicant details such as income type, annual income, employment duration, education level, housing type, loan balance, credit inquiries, and payment history.
- Cleans and prepares the data for machine learning.
- Trains multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- Compares model performance using accuracy, precision, recall, F1 score, confusion matrix, and classification report.
- Saves the best-performing model.
- Applies practical credit-policy guardrails so clearly ineligible profiles, such as extremely low income or debt greater than income, are rejected even if the model probability is uncertain.
- Runs a Flask web app for real-time approval prediction.
- Provides a batch prediction script for screening multiple applicants from a CSV file.
- Includes an optional IBM Watson Machine Learning deployment script.

## Project Structure

```text
Credit_Card/
 app.py
 credit_card_approval/
   config.py
   data.py
   features.py
   modeling.py
   prediction.py
   schemas.py
   visualization.py
 scripts/
   batch_predict.py
   run_eda.py
   train_model.py
   watson_deploy.py
 templates/
 static/
 data/
   raw/
   processed/
 models/
 reports/
   figures/
 requirements.txt
 requirements-watson.txt
 README.md
```

## Requirements

Hardware:

- Intel i3 processor or above
- Minimum 4 GB RAM, 8 GB recommended
- At least 2 GB free disk space
- A modern web browser such as Chrome, Edge, or Firefox

Software:

- Python 3.8 or above
- VS Code, PyCharm, or any code editor
- Jupyter Notebook or Anaconda Navigator if you want to explore the dataset manually

## Setup

Open a terminal in the project folder and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If you are using Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

You can train the model with the built-in sample dataset:

```bash
python scripts/train_model.py
```

The project creates a realistic sample dataset automatically if no dataset is found. This makes the application runnable immediately.

To train with your own CSV file:

```bash
python scripts/train_model.py --data data/raw/your_dataset.csv
```

After training, the model is saved here:

```text
models/credit_card_approval_model.joblib
```

The model metrics are saved here:

```text
models/model_metrics.json
```

## Run the Flask Application

Start the web app:

```bash
python app.py
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

If the model has not been trained yet, the Flask app trains one automatically on first run.

## Generate Data Visualizations

Run this command:

```bash
python scripts/run_eda.py
```

Charts are saved in:

```text
reports/figures/
```

The generated charts include approval counts, income distribution, approval rate by income type, and numeric feature correlation.

## Batch Prediction

To screen many applicants from a CSV file:

```bash
python scripts/batch_predict.py --input data/raw/applicants.csv --output data/processed/predictions.csv
```

The output file will include the original applicant data plus:

- prediction decision
- approval probability
- risk level

## API Prediction

The Flask app also provides a JSON endpoint:

```text
POST /api/predict
```

Example request body:

```json
{
  "gender": "Female",
  "annual_income": 65000,
  "income_type": "Working",
  "employment_years": 5,
  "education_level": "Higher education",
  "family_status": "Married",
  "housing_type": "House / apartment",
  "age": 34,
  "children_count": 0,
  "family_members": 2,
  "own_car": "No",
  "own_property": "Yes",
  "existing_loan_balance": 8500,
  "credit_inquiries_6m": 1,
  "payment_status": "Clean history"
}
```

## Dataset Notes

The app supports a clean CSV with columns such as:

- gender
- own_car
- own_property
- children_count
- annual_income
- income_type
- education_level
- family_status
- housing_type
- employment_years
- age
- family_members
- existing_loan_balance
- credit_inquiries_6m
- payment_status
- approved

The target column is:

```text
approved
```

Use `1` for approved and `0` for rejected.

The code also understands common column names from popular credit card approval datasets, such as `CODE_GENDER`, `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, and `STATUS`.

## IBM Watson Machine Learning Deployment

Install the optional Watson dependency:

```bash
pip install -r requirements-watson.txt
```

Set these environment variables from your IBM Cloud Watson Machine Learning service and deployment space:

```text
IBM_CLOUD_API_KEY
IBM_WML_URL
IBM_WML_SPACE_ID
IBM_WML_SOFTWARE_SPEC
IBM_WML_MODEL_TYPE
```

Train the model first, then run:

```bash
python scripts/watson_deploy.py
```

The script uploads the saved model and creates an online deployment in IBM Watson Machine Learning.

## Important Notes

This project is built for education and demonstration. Real credit approval systems need stronger controls, including:

- verified production data
- fairness and bias testing
- explainability reports
- audit logging
- secure authentication
- privacy protection
- regulatory approval

The model should support human decision-making, not replace responsible financial review.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train_model.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```



