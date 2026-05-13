# ============================================================
# Assignment 1 - Data Wrangling I
# Dataset: Salary_dataset.csv (Source: https://www.kaggle.com)
# ============================================================

# STEP 1: Import all required Python Libraries
import pandas as pd
import numpy as np

# -------------------------------------------------------
# STEP 2: Load the Dataset into pandas DataFrame
# -------------------------------------------------------
data = pd.read_csv("Salary_dataset.csv")
print("\nDataset:")
print(data)

# Display column names
print("\nColumn Names:")
print(data.columns)

# -------------------------------------------------------
# STEP 3: Data Preprocessing
# -------------------------------------------------------

# Check dimensions of the dataframe
print("\nShape (rows, columns):", data.shape)

# Check data types of each column
print("\nData Types:")
print(data.dtypes)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Get initial statistics using describe()
print("\nStatistical Description:")
print(data.describe())

# Display first 5 rows
print("\nHead (first 5 rows):")
print(data.head())

# Display last 5 rows
print("\nTail (last 5 rows):")
print(data.tail())

# Detailed info of the dataframe
print("\nDataFrame Info:")
print(data.info())

# -------------------------------------------------------
# STEP 4: Data Formatting and Type Conversions
# -------------------------------------------------------

# Drop the unnamed index column
data = pd.read_csv("Salary_dataset.csv", index_col=0)

print("\nData Types before conversion:")
print(data.dtypes)

# YearsExperience is float64 - correct
# Salary is float64 - correct
# Convert Salary to int for cleaner representation
data['Salary'] = data['Salary'].astype('int64')

print("\nData Types after conversion:")
print(data.dtypes)

print("\nData after type conversion:")
print(data.head())

# -------------------------------------------------------
# STEP 5: Handle Missing Values
# -------------------------------------------------------

# Check missing values
print("\nMissing values count:")
print(data.isnull().sum())

# Fill missing values with mean (if any)
data['YearsExperience'] = data['YearsExperience'].fillna(data['YearsExperience'].mean())
data['Salary'] = data['Salary'].fillna(data['Salary'].mean())

print("\nMissing values after filling:")
print(data.isnull().sum())

# -------------------------------------------------------
# STEP 6: Turn Categorical Variables into Quantitative
# (from ass1.ipynb - using the dictionary/DataFrame example)
# -------------------------------------------------------

# Example from ass1.ipynb - Student marks dictionary
di = {
    'Roll':  [2, 4, 8, 9, 10],
    'Name':  ['Vivek', 'Aboli', 'Shrikant', 'Sita', 'Vijay'],
    'Marks': ['First', 'Distinction', 'Distinction', 'Second', 'First']
}
df = pd.DataFrame(di)
print("\nOriginal DataFrame:")
print(df)

print("\nData Types:")
print(df.dtypes)

# Type conversion - Roll to int32, Name to string
df = df.astype({'Name': 'string', 'Roll': 'int32'})
print("\nAfter type conversion:")
print(df.dtypes)

# Convert categorical Marks to numeric using replace
df['Marks'] = df['Marks'].replace(['Distinction', 'First', 'Second'], [0, 1, 2])
print("\nAfter replacing Marks with numeric values:")
print(df)

# Encoding using cat.codes
df["Marks"] = df["Marks"].astype('category')
df["Marks"] = df["Marks"].cat.codes
print("\nAfter encoding Marks using cat.codes:")
print(df)
print("\nFinal Data Types:")
print(df.dtypes)

# -------------------------------------------------------
# STEP 7: Data Normalization on Salary Dataset
# -------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['YearsExperience', 'Salary']] = scaler.fit_transform(
    data[['YearsExperience', 'Salary']]
)
print("\nNormalized Data (MinMaxScaler):")
print(data.head(10))





#OUTPUT
# Dataset:
#     Unnamed: 0  YearsExperience    Salary
# 0            0              1.2   39344.0
# 1            1              1.4   46206.0
# 2            2              1.6   37732.0
# 3            3              2.1   43526.0
# 4            4              2.3   39892.0
# 5            5              3.0   56643.0
# 6            6              3.1   60151.0
# 7            7              3.3   54446.0
# 8            8              3.3   64446.0
# 9            9              3.8   57190.0
# 10          10              4.0   63219.0
# 11          11              4.1   55795.0
# 12          12              4.1   56958.0
# 13          13              4.2   57082.0
# 14          14              4.6   61112.0
# 15          15              5.0   67939.0
# 16          16              5.2   66030.0
# 17          17              5.4   83089.0
# 18          18              6.0   81364.0
# 19          19              6.1   93941.0
# 20          20              6.9   91739.0
# 21          21              7.2   98274.0
# 22          22              8.0  101303.0
# 23          23              8.3  113813.0
# 24          24              8.8  109432.0
# 25          25              9.1  105583.0
# 26          26              9.6  116970.0
# 27          27              9.7  112636.0
# 28          28             10.4  122392.0
# 29          29             10.6  121873.0

# Column Names:
# Index(['Unnamed: 0', 'YearsExperience', 'Salary'], dtype='str')

# Shape (rows, columns): (30, 3)

# Data Types:
# Unnamed: 0           int64
# YearsExperience    float64
# Salary             float64
# dtype: object

# Missing Values:
# Unnamed: 0         0
# YearsExperience    0
# Salary             0
# dtype: int64

# Statistical Description:
#        Unnamed: 0  YearsExperience         Salary
# count   30.000000        30.000000      30.000000
# mean    14.500000         5.413333   76004.000000
# std      8.803408         2.837888   27414.429785
# min      0.000000         1.200000   37732.000000
# 25%      7.250000         3.300000   56721.750000
# 50%     14.500000         4.800000   65238.000000
# 75%     21.750000         7.800000  100545.750000
# max     29.000000        10.600000  122392.000000

# Head (first 5 rows):
#    Unnamed: 0  YearsExperience   Salary
# 0           0              1.2  39344.0
# 1           1              1.4  46206.0
# 2           2              1.6  37732.0
# 3           3              2.1  43526.0
# 4           4              2.3  39892.0

# Tail (last 5 rows):
#     Unnamed: 0  YearsExperience    Salary
# 25          25              9.1  105583.0
# 26          26              9.6  116970.0
# 27          27              9.7  112636.0
# 28          28             10.4  122392.0
# 29          29             10.6  121873.0

# DataFrame Info:
# <class 'pandas.DataFrame'>
# RangeIndex: 30 entries, 0 to 29
# Data columns (total 3 columns):
#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   Unnamed: 0       30 non-null     int64  
#  1   YearsExperience  30 non-null     float64
#  2   Salary           30 non-null     float64
# dtypes: float64(2), int64(1)
# memory usage: 852.0 bytes
# None

# Data Types before conversion:
# YearsExperience    float64
# Salary             float64
# dtype: object

# Data Types after conversion:
# YearsExperience    float64
# Salary               int64
# dtype: object

# Data after type conversion:
#    YearsExperience  Salary
# 0              1.2   39344
# 1              1.4   46206
# 2              1.6   37732
# 3              2.1   43526
# 4              2.3   39892

# Missing values count:
# YearsExperience    0
# Salary             0
# dtype: int64

# Missing values after filling:
# YearsExperience    0
# Salary             0
# dtype: int64

# Original DataFrame:
#    Roll      Name        Marks
# 0     2     Vivek        First
# 1     4     Aboli  Distinction
# 2     8  Shrikant  Distinction
# 3     9      Sita       Second
# 4    10     Vijay        First

# Data Types:
# Roll     int64
# Name       str
# Marks      str
# dtype: object

# After type conversion:
# Roll      int32
# Name     string
# Marks       str
# dtype: object

# After replacing Marks with numeric values:
#    Roll      Name Marks
# 0     2     Vivek     1
# 1     4     Aboli     0
# 2     8  Shrikant     0
# 3     9      Sita     2
# 4    10     Vijay     1

# After encoding Marks using cat.codes:
#    Roll      Name  Marks
# 0     2     Vivek      1
# 1     4     Aboli      0
# 2     8  Shrikant      0
# 3     9      Sita      2
# 4    10     Vijay      1

# Final Data Types:
# Roll      int32
# Name     string
# Marks      int8
# dtype: object

# Normalized Data (MinMaxScaler):
#    YearsExperience    Salary
# 0         0.000000  0.019041
# 1         0.021277  0.100094
# 2         0.042553  0.000000
# 3         0.095745  0.068438
# 4         0.117021  0.025514
# 5         0.191489  0.223376
# 6         0.202128  0.264812
# 7         0.223404  0.197425
# 8         0.223404  0.315545
# 9         0.276596  0.229837
# PS C:\Users\PRERANA\OneDrive\Desktop\practice\assignment1> 