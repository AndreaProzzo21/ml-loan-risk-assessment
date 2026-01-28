# Loan Approval Classification & Risk Score Regression

This repository contains a comprehensive Data Science project focused on predicting loan approval outcomes and estimating customer credit risk scores using demographic and financial data.

## 📊 Data Source
The dataset used in this project was sourced from **Kaggle**. 
* **Dataset Name**: Financial Risk for Loan Approval
* **Source**: https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval

Special thanks to the author for providing this data for public use and analysis.

## 📌 Project Overview

The project addresses two main financial challenges:

1. **Binary Classification**: Predicting whether a loan application will be `Approved` or `Denied`.
2. **Regression Analysis**: Estimating a continuous `Risk Score` for each client to assess creditworthiness.

By comparing multiple Machine Learning models, this project identifies the most effective algorithms for financial risk assessment and highlights the key features influencing bank decisions.

## 📊 Key Results & Insights

Based on the analysis of 2,000 client records (as detailed in the included Project Report):

### Classification (Loan Approval)

We compared Decision Trees, Random Forests, and Support Vector Machines (SVM).

* **Top Performer**: **SVM** (Linear Kernel).
* **Accuracy**: ~96.7% on cross-validation.
* **Generalization**: The model maintained high performance (96.2% accuracy) on unseen data, indicating no significant overfitting.

### Regression (Risk Score)

* **Algorithm**: Linear Regression.
* **Performance**: Achieved a strong  score, indicating that financial features (Income, DTI, etc.) are highly predictive of the calculated risk score.

### Feature Importance (Pearson Correlation)

* **Positive Drivers**: Monthly/Annual Income and Age are the strongest indicators of approval.
* **Negative Drivers**: Debt-to-Income (DTI) Ratio and the Risk Score itself are the primary reasons for loan rejection.
* **The "Outlier" Insight**: High Net Worth and low Loan Amounts can lead to approval even when the Risk Score is sub-optimal.

---

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Libraries**:
* `Scikit-Learn`: Model training, cross-validation, and metrics.
* `Pandas/NumPy`: Data manipulation and encoding.
* `Matplotlib/Seaborn`: Visualizing correlations and model performance.



---

## 📂 Project Structure

* `data/`: Contains the `demograpich_financial_dataset.csv`.
* `plots/`: Automatically generated visualizations (Correlation heatmaps, Model comparisons).
* `src/`: The modular Python script (`LoanProject` class).
* `Report.pdf`: The full detailed technical report of the methodology.

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/AndreaProzzo21/.git

```


2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```


3. Run the main script:
```bash
python src/main_pipeline.py

```



---

## 📖 Methodology Summary

The project follows a rigorous pipeline:

1. **Preprocessing**: Handling string formats and categorical encoding via `get_dummies`.
2. **Data Scaling**: Normalizing features using `StandardScaler` to ensure model stability.
3. **Model Arena**: Using `cross_validate` to compare different architectures.
4. **Hyperparameter Tuning**: Optimizing the SVM via `GridSearchCV` (, linear kernel).
