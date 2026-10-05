# Credit Card Fraud Detection

Machine learning project (CODESOFT Task 2) that classifies credit card transactions as **LEGITIMATE** or **FRAUDULENT**, with a Flask web app.

**Stack:** Python, Pandas, NumPy, Scikit-learn, Joblib, Flask, HTML, CSS, JavaScript

## Dataset
Columns come from the uploaded files (`fraudTrain.csv`, `fraudTest.csv`).
- **Target:** `is_fraud` (0 = legitimate, 1 = fraud)
- **Used columns:** `trans_date_trans_time`, `amt`, `category`, `gender`, `state`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long`, `dob`
- **Engineered features** (in `preprocessing.py`): `hour`, `day_of_week`, `age`, `distance_km` (customer ↔ merchant)
- **Not used:** ID/personal columns (`Unnamed: 0`, `cc_num`, `first`, `last`, `street`, `trans_num`) and high-cardinality columns (`merchant`, `city`, `zip`, `job`, `unix_time`)

## Project structure
```
credit-card-fraud-detection/
├── app.py                 # Flask app + POST /predict
├── preprocessing.py       # shared preprocessing (training + prediction)
├── requirements.txt
├── README.md
├── .gitignore
├── data/                  # put fraudTrain.csv and fraudTest.csv here
├── model/                 # fraud_model.pkl, metrics.json, options.json
├── training/train_model.py
├── templates/index.html
└── static/css/style.css, static/js/script.js
```

## Setup (VS Code)
1. Open the `credit-card-fraud-detection` folder in VS Code.
2. Copy `fraudTrain.csv` and `fraudTest.csv` into the `data/` folder.
3. Open a terminal and run:
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python training/train_model.py   # trains models, prints comparison, saves model
python app.py                    # starts the app
```
4. Open http://127.0.0.1:5000

## How it works
- **Preprocessing:** date/time feature engineering, median imputing + scaling for numbers, most-frequent imputing + one-hot encoding for categories. Unknown categories are ignored safely.
- **Models:** Logistic Regression, Decision Tree, Random Forest.
- **Class imbalance:** fraud is only about 0.6% of the training rows, so every model uses `class_weight` (balanced).
- **Training size:** a stratified sample of 300,000 training rows is used for speed. Change `MAX_TRAIN_ROWS` in `training/train_model.py` (`None` = all rows).
- **Evaluation:** all models are scored on `fraudTest.csv` using Accuracy, Precision, Recall, F1 and ROC-AUC. Accuracy alone is misleading on imbalanced data, so the best model is chosen by **F1 score**.
- **Saved model:** one Joblib pipeline (preprocessing + model) in `model/fraud_model.pkl`. `app.py` loads the same pipeline, so prediction uses the same preprocessing as training.

## API
`POST /predict` with JSON containing the 11 used columns:
```json
{
  "trans_date_trans_time": "2020-06-21T12:14", "amt": 2.86,
  "category": "personal_care", "gender": "M", "state": "SC",
  "city_pop": 333497, "lat": 33.9659, "long": -80.9355,
  "merch_lat": 33.986391, "merch_long": -81.200714, "dob": "1968-03-19"
}
```
Response: `{"prediction": "LEGITIMATE", "fraud_probability": 0.1072}`

## Results
Run `python training/train_model.py` to see the comparison table on your machine. Metrics are also saved to `model/metrics.json`.
