# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('house_price.csv')

# defining X and Y

X = dataset.iloc[:, :-1].values # X is taking all rows, first column (index0)
y = dataset.iloc[:, -1].values # .values is a built in method to convert dataframe into array

# We can also convert pandas dataframe to numpy array using .to_numpy():
# my_array = dataset.to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Area (Training set)')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Area (Test set)')
plt.xlabel('Area of the house')
plt.ylabel('House price')
plt.show()


# Model Intercept and Slope
intercept = regressor.intercept_
slope = regressor.coef_

print ("slope is: ", slope, " and intercept is: " , intercept)

# R-squared
r_squared = regressor.score(X, y)
print ("R_squared is: ", r_squared)


