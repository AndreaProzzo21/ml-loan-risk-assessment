import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

class LoanProject:
    def __init__(self, filepath, output_dir='plots'):
        self.raw_data = pd.read_csv(filepath)
        self.scaler = StandardScaler()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.df = self._clean_and_encode()

    def _clean_and_encode(self):
        """Preprocessing iniziale e encoding."""
        df = self.raw_data.copy()
        df.columns = df.columns.str.strip()
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        categorical_cols = ['MaritalStatus','HomeOwnershipStatus','EducationLevel',
                            'EmploymentStatus','LoanPurpose']
        return pd.get_dummies(df.drop('ApplicationDate', axis=1), columns=categorical_cols)

    def get_task_data(self, target_column):
        """Separa X e y in base al task."""
        X = self.df.drop(['LoanApproved', 'RiskScore'], axis=1)
        y = self.df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def compare_classifiers(self, X_train, y_train):
        """Model Arena per la classificazione con plot dei risultati."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        models = {
            'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
            'SVM': SVC(kernel='linear', C=1)
        }
        
        results = {}
        for name, model in models.items():
            cv = cross_validate(model, X_train_scaled, y_train, cv=5, 
                                scoring=['accuracy', 'precision', 'recall'])
            results[name] = {
                'Accuracy': cv['test_accuracy'].mean(),
                'Precision': cv['test_precision'].mean(),
                'Recall': cv['test_recall'].mean()
            }
        
        df_results = pd.DataFrame(results).T
        self._plot_classifier_comparison(df_results)
        return df_results

    def run_regression(self, X_train, X_test, y_train, y_test):
        """Esegue regressione e genera il plot di confronto real vs pred."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        regressor = LinearRegression()
        regressor.fit(X_train_scaled, y_train)
        preds = regressor.predict(X_test_scaled)
        
        self._plot_regression_results(y_test, preds)
        
        return {
            'RMSE': root_mean_squared_error(y_test, preds),
            'R2_Score': r2_score(y_test, preds)
        }

    # --- METODI DI VISUALIZZAZIONE ---

    def _plot_classifier_comparison(self, df_results):
        plt.figure(figsize=(10, 6))
        df_results.plot(kind='bar', ax=plt.gca())
        plt.title('Classifiers Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/classifier_comparison.png')
        plt.show()

    def _plot_regression_results(self, y_true, y_pred):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Risk Score')
        plt.ylabel('Predicted Risk Score')
        plt.title('Regression: Actual vs Predicted Risk Score')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/regression_fit.png')
        plt.show()

    def plot_correlation_matrix(self):
        """Genera una heatmap delle correlazioni per i fattori principali."""
        features = ['AnnualIncome', 'MonthlyDebtPayments', 'DebtToIncomeRatio', 
                    'RiskScore', 'LoanApproved', 'Age']
        # Filtriamo solo le colonne presenti nel df finale
        cols = [c for c in features if c in self.df.columns]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df[cols].corr(), annot=True, cmap='RdYlGn', fmt=".2f")
        plt.title('Key Financial Factors Correlation')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png')
        plt.show()

# --- ESECUZIONE ---
project = LoanProject("data/demographic_financial_dataset.csv")

# 0. Analisi Correlazione (Insights per il README)
project.plot_correlation_matrix()

# 1. CLASSIFICAZIONE
X_train_c, X_test_c, y_train_c, y_test_c = project.get_task_data('LoanApproved')
print("\n--- Model Arena (Classification) ---")
print(project.compare_classifiers(X_train_c, y_train_c))

# 2. REGRESSIONE
X_train_r, X_test_r, y_train_r, y_test_r = project.get_task_data('RiskScore')
print("\n--- Regression Performance (Risk Score) ---")
print(project.run_regression(X_train_r, X_test_r, y_train_r, y_test_r))