# Name: Nilesh Desai
# Virtual Internship Assignment Week 1
# Topic: Data Preprocessing using Pandas and Scikit-learn
# Dataset: Titanic (tested.csv)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Loading the Titanic dataset into a dataframe
df = pd.read_csv('tested.csv')

# First lets see how many rows and columns we have
print("Shape of dataset:", df.shape)

# .info() gives us column names, data types and non-null counts
# this helps us understand the structure of our data
print("\n--- Dataset Info ---")
print(df.info())

# .describe() gives statistical summary like mean, min, max etc
# only works on numerical columns by default
print("\n--- Statistical Summary ---")
print(df.describe())

# lets also look at first 5 rows to understand the data visually
print("\n--- First 5 Rows ---")
print(df.head())

# Before doing anything lets check which columns have missing values
# and how many are missing in each column
print("\n--- Missing Values BEFORE Cleaning ---")
print(df.isnull().sum())

# --- Age Column ---
# Age has some missing values
# we will fill them using MEAN because age is normally distributed
age_mean = df['Age'].mean()
print(f"\nFilling Age missing values with Mean: {age_mean:.2f}")
df['Age'].fillna(age_mean, inplace=True)

# --- Fare Column ---
# Fare can have outliers so MEDIAN is a better choice here
# median is not affected by extreme values like mean is
fare_median = df['Fare'].median()
print(f"Filling Fare missing values with Median: {fare_median:.2f}")
df['Fare'].fillna(fare_median, inplace=True)

# --- Embarked Column ---
# Embarked is a categorical column
# we use MODE which means the most frequently occurring value
if 'Embarked' in df.columns:
    embarked_mode = df['Embarked'].mode()[0]
    print(f"Filling Embarked missing values with Mode: {embarked_mode}")
    df['Embarked'].fillna(embarked_mode, inplace=True)

# --- Cabin Column ---
# Cabin has too many missing values (over 70%)
# it is better to drop it completely rather than imputing
if 'Cabin' in df.columns:
    print("\nDropping Cabin column as it has too many missing values")
    df.drop(columns=['Cabin'], inplace=True)

# Now lets verify all missing values are handled
print("\n--- Missing Values AFTER Cleaning ---")
print(df.isnull().sum())

# Machine learning models cannot work with text data directly
# so we need to convert text columns into numbers

# --- Label Encoding on Sex column ---
# LabelEncoder converts categories into 0 and 1
# male = 1, female = 0 (alphabetical order)
print("\n--- Label Encoding on Sex column ---")
le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['Sex'])
print(df[['Sex', 'Sex_Encoded']].head(10))

# --- OneHot Encoding on Embarked column ---
# OneHotEncoding creates a separate column for each category
# this avoids giving any false ranking between categories
# for example we dont want model to think S > Q > C just because of numbers
print("\n--- OneHot Encoding on Embarked column ---")
if 'Embarked' in df.columns:
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    print(df.filter(like='Embarked').head(10))

print("\n========================================")
print("   MINI PROJECT: TITANIC DATA CLEANING  ")
print("========================================")

# Overview of what we did
print("\nSteps performed in this project:")
print("1. Loaded the Titanic dataset")
print("2. Explored the data using info() and describe()")
print("3. Identified missing values in Age, Fare, Cabin, Embarked")
print("4. Filled Age with Mean, Fare with Median, Embarked with Mode")
print("5. Dropped Cabin due to excessive missing values")
print("6. Applied LabelEncoder on Sex column")
print("7. Applied OneHotEncoder on Embarked column")

# Final shape of cleaned dataset
print(f"\nOriginal columns we worked with: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked")
print(f"\nFinal Dataset Shape after cleaning: {df.shape}")

# Final info of cleaned data
print("\n--- Final Cleaned Dataset Info ---")
print(df.info())

# Final statistical summary
print("\n--- Final Statistical Summary ---")
print(df.describe())

# Saving the cleaned dataset to a new CSV file
df.to_csv('titanic_cleaned.csv', index=False)
print("\n Cleaned dataset has been saved as 'titanic_cleaned.csv'")
print(" Data preprocessing completed successfully!")