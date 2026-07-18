=========================================
HOUSE PRICE PREDICTION USING LINEAR REGRESSION
=========================================

Project Description
-------------------
This project predicts house prices using the Linear Regression algorithm.
The model is trained on a housing dataset and predicts the price of a house
based on features such as area, number of bedrooms, bathrooms, stories,
parking, and other available attributes.

Objectives
----------
1. Load a housing dataset.
2. Clean and preprocess the data.
3. Explore the dataset using graphs and correlation analysis.
4. Select important features.
5. Split the dataset into training and testing sets.
6. Train a Linear Regression model.
7. Evaluate the model using RMSE and R² Score.
8. Interpret the model coefficients.
9. Save the trained model.
10. Predict house prices for new data.

Technologies Used
-----------------
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

Project Files
-------------
housing.csv                 -> Dataset
house_price_prediction.py   -> Main Python program
house_model.pkl             -> Saved trained model
README.txt                  -> Project documentation

How to Run
----------
1. Install the required libraries:

   pip install pandas numpy matplotlib seaborn scikit-learn joblib

2. Place housing.csv in the project folder.

3. Run the program:

   python house_price_prediction.py

Output
------
The program will:
- Load the dataset.
- Display basic dataset information.
- Clean missing values (if any).
- Show data visualizations.
- Train the Linear Regression model.
- Display RMSE and R² Score.
- Print model coefficients.
- Save the trained model as house_model.pkl.
- Predict the price of a sample house.
