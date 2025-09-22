# End-to-End ML Pipeline for Customer Churn Prediction

## Overview

This project implements a reusable and production-ready machine learning pipeline using Scikit-learn's Pipeline API to predict customer churn based on the Telco Customer Churn dataset. The pipeline includes data preprocessing (handling numerical scaling and categorical encoding), model training with Logistic Regression and Random Forest classifiers, hyperparameter tuning via GridSearchCV, and model export using Joblib for easy deployment.

Key features:
- Automated data cleaning and feature preparation.
- Hyperparameter optimization for improved model performance.
- Evaluation metrics including accuracy, precision, recall, and F1-score.
- Exportable pipeline for inference on new data.

This setup is ideal for demonstrating best practices in building scalable ML workflows and can be easily adapted for similar classification tasks.

## Dataset

The dataset used is the [Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) from IBM. It contains customer information such as demographics, account details, and services, with the target variable "Churn" indicating whether a customer left the service (Yes/No).

- Rows: 7,043
- Columns: 21 (including target)
- Features: Numerical (e.g., tenure, MonthlyCharges) and categorical (e.g., gender, InternetService).

Data is loaded directly from a GitHub URL in the code, with basic cleaning applied (e.g., handling invalid values in TotalCharges).

## Requirements

- Python 3.6+
- Libraries: 
  - pandas
  - numpy
  - scikit-learn
  - joblib

Install dependencies using:
```
pip install pandas numpy scikit-learn joblib
```

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Run the script:
   ```
   python churn_pipeline.py
   ```

   - The script will load the dataset, preprocess it, train and tune models, print evaluation results, and export the best Random Forest pipeline as `churn_pipeline_rf.pkl`.

3. Example output:
   - Best hyperparameters for each model.
   - Test set accuracy (typically ~0.79-0.80).
   - Classification report.

4. For inference on new data:
   ```python
   import joblib
   import pandas as pd

   # Load the pipeline
   loaded_pipeline = joblib.load('churn_pipeline_rf.pkl')

   # Prepare new data (same format as training data, excluding 'Churn' and 'customerID')
   new_data = pd.DataFrame({...})  # Your new customer data

   # Predict
   predictions = loaded_pipeline.predict(new_data)
   print(predictions)  # [0, 1, ...] where 1 = Churn
   ```

## Code Structure

- **Data Loading & Cleaning**: Handles dataset import and fixes common issues.
- **Preprocessing Pipeline**: Uses `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical ones.
- **Model Training & Tuning**: Defines a function to build pipelines, perform GridSearchCV, and evaluate models.
- **Models**: Logistic Regression and Random Forest with predefined hyperparameter grids.
- **Export**: Saves the tuned pipeline using Joblib.

The full code is in `churn_pipeline.py` (or inline in the Jupyter notebook if using Colab).

## Results

- Logistic Regression: Accuracy ~0.80 (varies with tuning).
- Random Forest: Accuracy ~0.79-0.80, often slightly better on imbalanced classes due to ensemble nature.

For detailed metrics, run the script.

## Contributing

Feel free to fork, submit issues, or pull requests. Suggestions for additional models (e.g., XGBoost) or features are welcome!

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Dataset provided by IBM.
- Built with Scikit-learn for efficient ML pipelining.

Last updated: September 22, 2025
