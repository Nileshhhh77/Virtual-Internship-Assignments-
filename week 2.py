# Name: Nilesh Desai
# Virtual Internship Assignment - Week 2
# Topic: Linear Regression & Logistic Regression
# Datasets: Housing.csv, Titanic (tested.csv)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score



print("========================================")
print("   TASK 1: HOUSE PRICE - LINEAR REGRESSION")
print("========================================")

house_df = pd.read_csv('Housing.csv')

print("\nDataset Shape:", house_df.shape)
print("\nFirst 5 rows:")
print(house_df.head())

print("\nMissing values:")
print(house_df.isnull().sum())

# This dataset has yes/no columns which need to be converted to numbers
# before we can use them in Linear Regression
yes_no_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

le = LabelEncoder()
for col in yes_no_cols:
    house_df[col] = le.fit_transform(house_df[col])  # yes=1, no=0

# furnishingstatus has 3 categories so we use OneHotEncoding
furnishing_dummies = pd.get_dummies(house_df['furnishingstatus'], prefix='furnishing')
house_df = pd.concat([house_df, furnishing_dummies], axis=1)
house_df.drop(columns=['furnishingstatus'], inplace=True)

print("\nDataset after encoding categorical columns:")
print(house_df.head())

# price is the target column we want to predict
X = house_df.drop(columns=['price'])
y = house_df['price']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = lr_model.predict(X_test)

# Evaluate using R2 score
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score for House Price Prediction: {r2:.4f}")

# Show some actual vs predicted values
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nActual vs Predicted (first 10 rows):")
print(results_df.head(10))


print("\n========================================")
print("   MINI PROJECT 2: HOUSE PRICE PREDICTION")
print("========================================")

print("\nSteps performed:")
print("1. Loaded Housing dataset")
print("2. Checked for missing values")
print("3. Encoded yes/no columns using LabelEncoder")
print("4. Encoded furnishingstatus using OneHotEncoder")
print("5. Split data into train (80%) and test (20%) sets")
print("6. Trained a Linear Regression model")
print("7. Predicted house prices on test data")
print(f"8. Evaluated model performance using R2 Score = {r2:.4f}")
print("9. Plotted Actual vs Predicted values")

# Plot Predicted vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('house_price_plot.png')
print("\n Plot saved as 'house_price_plot.png'")



print("\n========================================")
print("   TASK 2: TITANIC - LOGISTIC REGRESSION")
print("========================================")

titanic_df = pd.read_csv('tested.csv')

print("\nMissing values before cleaning:")
print(titanic_df.isnull().sum())

# Fill missing Age with mean
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

# Fill missing Fare with median
titanic_df['Fare'] = titanic_df['Fare'].fillna(titanic_df['Fare'].median())

# Fill missing Embarked with mode
if titanic_df['Embarked'].isnull().sum() > 0:
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

# Encode Sex column using LabelEncoder
le2 = LabelEncoder()
titanic_df['Sex_Encoded'] = le2.fit_transform(titanic_df['Sex'])

# Encode Embarked using OneHotEncoding
embarked_dummies = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
titanic_df = pd.concat([titanic_df, embarked_dummies], axis=1)

# Select features for Logistic Regression
log_features = ['Pclass', 'Sex_Encoded', 'Age', 'Fare', 'SibSp', 'Parch']
X_titanic = titanic_df[log_features]
y_titanic = titanic_df['Survived']

# Split into train and test
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_t, y_train_t)

# Predict on test data
y_pred_t = log_model.predict(X_test_t)

# Evaluate accuracy
accuracy = accuracy_score(y_test_t, y_pred_t)
print(f"\nLogistic Regression Accuracy on Titanic Survival Prediction: {accuracy:.4f}")

# Show some actual vs predicted survival values
titanic_results = pd.DataFrame({'Actual': y_test_t.values, 'Predicted': y_pred_t})
print("\nActual vs Predicted Survival (first 10 rows):")
print(titanic_results.head(10))

print("\n Week 2 Assignment Completed Successfully!")