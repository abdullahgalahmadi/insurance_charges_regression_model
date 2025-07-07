
#  Medical Charges Prediction – Linear Regression Project

This project demonstrates how to build and evaluate a linear regression model to predict individual medical insurance charges using Python.

##  Dataset
The dataset (`insurance.csv`) includes the following features:
- `age`: Age of the primary beneficiary
- `sex`: Gender of the insured person
- `bmi`: Body mass index
- `children`: Number of dependents covered by insurance
- `smoker`: Smoking status
- `region`: Residential area in the US
- `charges`: Medical costs billed by insurance

##  Exploratory Data Analysis (EDA)
- Visualized the distribution of medical charges
- Created pair plots to understand variable relationships
- Used Label Encoding for categorical variables
- Generated a correlation heatmap

##  Model Building
- Linear Regression using `scikit-learn`
- Training/testing split: 70% / 30%
- Trained model using features: age, sex, bmi, children, smoker, region

## Model Evaluation
- Actual vs. Predicted values plotted
- Residual analysis through histogram
- Metrics used:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)

## Tools & Technologies
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn

##  Outcome
A complete machine learning pipeline from preprocessing and visualization to training and evaluation. The model achieved solid predictive performance using linear regression.

