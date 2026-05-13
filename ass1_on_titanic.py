# Data Wrangling I - Titanic Dataset

# Step 1 : Import Libraries
import pandas as pd
import numpy as np

# Step 2 : Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Step 3 : Display Dataset
print("FIRST 5 ROWS")
print(df.head())

# Step 4 : Data Preprocessing
# Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Statistical Information
print("\nSTATISTICS")
print(df.describe())

# Dataset Information
print("\nDATASET INFO")
print(df.info())

# Shape
print("\nDATASET SHAPE")
print(df.shape)

# Column Names
print("\nCOLUMN NAMES")
print(df.columns)

# Step 5 : Data Types
print("\nDATA TYPES")
print(df.dtypes)

# Handle Missing Values
# Fill Age with Mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill Embarked with Mode
df['Embarked'] = df['Embarked'].fillna(
    df['Embarked'].mode()[0]
)

# Fill Cabin with Unknown
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Type Conversion
df['Age'] = df['Age'].astype(int)


# Data Normalization
df['Fare'] = (
    (df['Fare'] - df['Fare'].min()) /
    (df['Fare'].max() - df['Fare'].min())
)

print("\nNORMALIZED FARE")
print(df['Fare'].head())


# Convert Categorical Variables
# Convert Sex
df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
})

# Convert Embarked
df['Embarked'] = df['Embarked'].map({
    'S': 0,
    'C': 1,
    'Q': 2
})

# Final Dataset
print("\nUPDATED DATASET")
print(df.head())

# Final Missing Values Check
print("\nFINAL MISSING VALUES")
print(df.isnull().sum())

# Final Dataset Information
print("\nFINAL DATASET INFO")
print(df.info())