import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset loaded successfully.\n")
print("First 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print("\nNumerical features normalized.")

sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age and Fare (after scaling)")
plt.grid(True)
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

print(f"\nShape after removing outliers: {df.shape}")

df.to_csv('titanic_cleaned_by_vijay.csv', index=False)
print("\nCleaned dataset saved as 'titanic_cleaned_by_vijay.csv'")
