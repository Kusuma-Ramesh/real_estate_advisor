# Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

An ML-powered real estate analytics application built using Python, Scikit-learn, Streamlit and MLflow integration.

The application helps users explore property characteristics, assess investment potential and examine estimated future property values through interactive analytics.

## Project Overview

**Domain:** Real Estate / Investment / Financial Analytics

**Technologies:**
- Python
- Pandas and NumPy
- Scikit-learn
- Machine Learning
- Regression and Classification
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Streamlit
- MLflow
- Plotly

## Key Features

- Investment potential classification.
- Current property price estimation.
- Five-year property price projection.
- Interactive property exploration and filtering.
- Location-based market insights.
- EDA visualizations and correlation analysis.
- Model evaluation and feature importance.
- Methodology and limitations documentation.

## Application Pages

1. **Predict:** Evaluate property details and generate investment assessments.
2. **Explore Properties:** Explore and filter property listings.
3. **Market Insights:** Visualize market patterns and property analytics.
4. **Model Details:** Review model performance and evaluation metrics.
5. **Methodology & Limitations:** Understand the assumptions, modelling approach and limitations.

## Machine Learning Approach

### 1. Data Preprocessing
- Missing-value handling.
- Duplicate detection.
- Numerical feature preprocessing and scaling.
- Categorical feature encoding.
- Feature engineering.
- Data quality auditing.

### 2. Exploratory Data Analysis
The project includes 20 EDA visualizations covering:
- Property price and size distributions.
- Price per square foot analysis.
- State and city comparisons.
- Property age and BHK distributions.
- Feature correlations.
- Nearby infrastructure analysis.
- Furnishing and property characteristics.
- Ownership and availability patterns.
- Parking, amenities and transportation.

### 3. Classification

Predicts whether a property meets the project's predefined Good Investment criteria.

**Important:** The classification target is a rule-based proxy, not an observed real-world investment outcome.

### 4. Regression

Estimates current property prices using available property attributes.

### 5. Five-Year Price Projection

Provides scenario-based price projections using assumed appreciation rates and predefined adjustment rules.

These projections are assumption-based calculations, not validated future-price predictions.

## Model Evaluation

### Classification

The classifier achieved approximately 99.9% accuracy in reproducing the rule-based target.

This result does not establish the ability to predict actual investment success.

### Regression

The evaluated regression model achieved approximately:

- RMSE: ₹141.2 lakh
- R²: approximately 0.000

The model did not outperform the mean-price baseline.

Detailed evaluation results are available in `reports/evaluation_report.md`.

## Project Structure

```text
real_estate_advisor/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── DATA_DICTIONARY.md
│
├── eda_outputs/
│   ├── findings.md
│   └── EDA visualizations
│
├── models/
│   ├── classifier.joblib
│   ├── regressor.joblib
│   └── metadata.json
│
├── reports/
│   ├── data_audit.md
│   ├── evaluation_report.md
│   ├── metrics.json
│   └── run_log.jsonl
│
├── src/
│   ├── audit.py
│   ├── eda.py
│   ├── features.py
│   ├── predict.py
│   ├── tracking.py
│   ├── train.py
│   └── ui.py
│
├── tests/
├── app.py
├── requirements.txt
├── smoke_test.py
├── .gitignore
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Kusuma-Ramesh/real_estate_advisor.git
cd real_estate_advisor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure input data

Input data files are intentionally not included in this public repository.

Provide an authorized, compatible local data source before running data-dependent operations.

### 5. Run the application

```bash
streamlit run app.py
```

### 6. Run tests

```bash
python -m pytest -v
```

### 7. Run EDA and data audit

```bash
python -m src.eda
python -m src.audit
```

### 8. Train the models

```bash
python -m src.train
```

MLflow tracking integration is provided in the training pipeline. Experiment history and registered models are not included as published tracking artifacts.

## Documentation

Additional project documentation:

- `data/DATA_DICTIONARY.md`
- `eda_outputs/findings.md`
- `reports/data_audit.md`
- `reports/evaluation_report.md`

## Limitations

- The classification target is based on predefined rules rather than observed investment outcomes.
- Future property prices are estimated using assumed appreciation scenarios.
- The regression model has limited predictive performance and does not outperform the mean baseline.
- Property characteristics and relationships in the underlying data have limitations.
- Reliable investment forecasting requires representative historical property transactions, resale prices and rental-return data.
- Model outputs should be treated as screening aids, not financial advice.

## Testing

The project has undergone:
- Automated unit testing.
- Python compilation checks.
- Model evaluation.
- Data auditing.
- Manual Streamlit application testing.

The locally tested application passed all 26 automated tests.

## Disclaimer

This application is developed for educational and analytical purposes. Its outputs are not financial, legal or investment advice. Users should independently verify property information and consult qualified professionals before making investment decisions.

## Author

**Kusuma Ramesh**

GitHub: https://github.com/Kusuma-Ramesh

---

*Built with Python, Machine Learning and Streamlit.*