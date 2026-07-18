import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib
df = pd.read_csv("housing.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df = df.dropna()
df.drop_duplicates(inplace=True)
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()
plt.show()
print(df.dtypes)
plt.scatter(df["area"], df["price"])
plt.xlabel("area")
plt.ylabel("price")
plt.show()
df.hist(figsize=(12,10))
plt.show()
X = df[["area","bedrooms","bathrooms"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
print(model.coef_)
print(model.intercept_)
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)
joblib.dump(model, "house_model.pkl")