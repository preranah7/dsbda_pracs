# Pima Diabetes Dataset

# Step 1 : Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Step 2 : Load Dataset
# Read CSV file
df = pd.read_csv("diabetes.csv")

# Step 3 : Display Dataset
print("\nFIRST 5 ROWS OF DATASET")
print(df.head())

# Step 4 : Dataset Information
print("\nDATASET INFORMATION")
print(df.info())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)


# Step 5 : Statistical Information
print("\nSTATISTICAL INFORMATION")
print(df.describe())

# Step 6 : Check Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 7 : Replace Invalid Zero Values with NaN
# In medical dataset, zero values are invalid
columns = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]
for col in columns:
    df[col] = df[col].replace(0, np.nan)

print("\nMISSING VALUES AFTER REPLACING 0")
print(df.isnull().sum())

# Step 8 : Fill Missing Values with Mean
for col in columns:
    df[col] = df[col].fillna(df[col].mean())

print("\nMISSING VALUES AFTER FILLING")
print(df.isnull().sum())

# Step 9 : Detect Outliers using Z-Score
print("\nOUTLIER DETECTION")
z = np.abs(stats.zscore(df))
outliers = np.where(z > 3)
print("OUTLIER ROW INDEXES")
print(outliers)

# Step 10 : Remove Outliers
df_clean = df[(z < 3).all(axis=1)]
print("\nSHAPE AFTER REMOVING OUTLIERS")
print(df_clean.shape)

# Step 11 : Correlation Matrix
print("\nCORRELATION MATRIX")
correlation = df_clean.corr()
print(correlation)

# Step 12 : Discretization / Binning
# Create Age Groups
df_clean['Age_Group'] = pd.cut(
    df_clean['Age'],
    bins=[20, 30, 40, 50, 60, 100],
    labels=[
        '20-30',
        '31-40',
        '41-50',
        '51-60',
        '60+'
    ]
)
print("\nAGE GROUPS")
print(df_clean[['Age', 'Age_Group']].head())

# Step 13 : Normalization using MinMaxScaler
scaler = MinMaxScaler()

numeric_columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

df_clean[numeric_columns] = scaler.fit_transform(
    df_clean[numeric_columns]
)

print("\nNORMALIZED DATA")
print(df_clean.head())

# Final Output
print("\nFINAL CLEANED DATASET")
print(df_clean.head())