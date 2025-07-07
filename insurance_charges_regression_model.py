import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("insurance.csv")
print(df.head())
print(df.info())
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["charges"], kde=True, bins=30)
plt.title('Distribution of Medical Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()


sns.pairplot(df, x_vars=['age', 'bmi', 'children'], y_vars='charges', kind='scatter', height=4)
plt.show()

# Convert non-numeric columns (sex, smoker, region) to numbers by using LabelEncoder and then merging them with a DataFrame.
df_encoded = df.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_encoded['sex'] = le.fit_transform(df['sex'])
df_encoded['smoker'] = le.fit_transform(df['smoker'])
df_encoded['region'] = le.fit_transform(df['region'])

correlation = df_encoded.corr()


plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Training abd Testing Split

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Model Training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")

# Model Testing
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred = model.predict(X_test)

print("Predicted Charges:")
print(y_pred[:5])  

print("Actual Charges:")
print(y_test.values[:5]) 

# scatter to show real values with prediction values

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs. Predicted Charges")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # خط مثالي
plt.show()

# Histogram to show real values with prediction values
residuals = y_test - y_pred

plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.show()

# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")





