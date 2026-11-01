# Predicting Loan Default Risk Using Mutual Information and Random Forest

An Artificial Intelligence course project that explores whether a borrower is likely to experience serious delinquency within the next two years. The task is treated as a **binary classification problem**:

- `0` — no serious delinquency
- `1` — serious delinquency

The project uses Mutual Information for feature selection and compares Logistic Regression with a Random Forest classifier.

## Project files

```text
Artificial-Intelligence-Final-Project/
├── B27294.pdf
├── cs-training.csv
├── final.ipynb
└── README.md
```

| File | Description |
|---|---|
| `B27294.pdf` | Project report and documentation |
| `cs-training.csv` | Credit-risk dataset used by the notebook |
| `final.ipynb` | Data preparation, analysis, feature selection, model training and evaluation |

## Technologies

- Python 3
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- joblib

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Ehmahm/ArtificialIntelligence-Final-Project.git
cd ArtificialIntelligence-Final-Project
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Set up Python

It is recommended to use a virtual environment.

**Windows**

```bash
py -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `final.ipynb` and run the cells from top to bottom. Keep `cs-training.csv` in the same directory as the notebook because the notebook loads it using:

```python
pd.read_csv("cs-training.csv")
```

## Project workflow

1. **Data loading and inspection** — load the credit-risk dataset and inspect its structure.
2. **Data preprocessing** — rename selected columns, remove the index column and fill missing income and dependent-count values with their medians.
3. **Exploratory data analysis** — examine the target distribution, missing values and feature correlations.
4. **Feature selection** — use `mutual_info_classif` to rank predictors and retain the ten highest-scoring features.
5. **Model development** — split the data into training and test sets, then train Logistic Regression and Random Forest models.
6. **Evaluation** — compare accuracy, precision, recall, F1-score, confusion matrices and ROC-AUC. The notebook also includes stratified five-fold cross-validation.
7. **Feature importance** — inspect the features contributing most to the Random Forest predictions.
8. **Model export** — save the fitted Random Forest model and scaler as `loan_default_model.joblib` and `scaler.joblib`.

## Dataset and target

The notebook uses the `cs-training.csv` dataset. Its target column is `SeriousDlqin2yrs`, renamed to `default` during preprocessing. The target indicates whether a person experienced serious delinquency within two years.

The dataset contains borrower and credit-history variables such as age, monthly income, debt ratio, revolving-credit utilisation, late-payment history, open credit lines and number of dependents.


## Reported results

The project report reports that Random Forest performed better than Logistic Regression, with a reported accuracy of **0.82** and mean ROC-AUC of **0.78**. These are the figures stated in the report; rerun the notebook to verify the results in your current environment.

## Outputs

After the notebook runs successfully, it saves:

- `loan_default_model.joblib` — trained Random Forest classifier
- `scaler.joblib` — fitted feature scaler

These files are generated when the model-saving cell is executed. They are not included in the repository by default.

## Author
**Magomu Emmanuel**  
BSc Data Science  
Artificial Intelligence — CSC2108

## Disclaimer

This is an academic machine-learning project. Its predictions are for educational and experimental purposes only. They should not be used as the sole basis for real lending or credit decisions.
