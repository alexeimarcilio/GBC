# -*- coding: utf-8 -*-
"""
###############################################################################

              ____  Multiple Linear Regression  ___ 
              
Week of    :       November 14th, 2020 
Prepared by:       Moe Fadae & Najem Bouazza

###############################################################################
"""
# -----------------------------------------------------------------------------
#*******************   STEP 1 : Libraries *************************************
# -----------------------------------------------------------------------------

import numpy as np  # library for scientific computing
import pandas as pd # data structures and data analysis package
import matplotlib.pyplot as plt # 2D plotting library


# -----------------------------------------------------------------------------
#*******************STEP 2 : Dataset - Import, Visualize and Process **********
# -----------------------------------------------------------------------------

df = pd.read_csv('house_price_data.csv')
dataset = df.copy() # We take a copy of our raw data, In order 
                    #  to avoid any loss of data (in case we do a modification
                    # and we try to cancel and go back)

#T0 check the first rows of our dataframe to have an idea about the data
dataset.head()

#get the dataset's columns
columns_list = dataset.columns

# Select some columns : Features and the label
dataset = dataset[["LotFrontage","LotArea","GarageCars","GarageArea","SalePrice"]]
print(dataset)

# Check for missing values in the whole dataframe
dataset.isnull().values.any()

# Returns the number of sampls with missing values
dataset.isnull().sum()

# impute the NaN with the column's mean
# If you don't force inplace to True, the modification won't take place
dataset.fillna(value=dataset["LotFrontage"].mean(), inplace = True)

# Double check if the missing values has been imputed.
dataset.isnull().sum()
dataset.head(5)

# Shows the statistics of the dataset
dataset.describe() 

#Checking the shape of our dataset : number of rows and columns
dataset.shape

# Explore the data -  Data visualization using seaborn library
import seaborn as sns


# Distribution plot of the column "SalePrice"
plt.figure() # to avoid getting the graphs one on top of the other
sns.distplot(dataset["SalePrice"], color = "red")

# Define X and y and convert from dataframe into array
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Check the correlation between columns (variables) 
Corr = dataset.corr()


#and plot it as a heatmap to visualize wher ethe correlation is high
plt.figure()
sns.heatmap(dataset.corr(), annot=True, cmap = "coolwarm")


#Check the distributions for variables 2 by 2
plt.figure()
sns.pairplot(dataset, hue = "GarageCars")


#Scale / Standardize the features:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# -----------------------------------------------------------------------------
#*******************  STEP 3 : Training set and Test set **********************
# -----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# -----------------------------------------------------------------------------
#*******************STEP 4 : Create and Train the Model
# -----------------------------------------------------------------------------


from sklearn.linear_model import LinearRegression


#Create the model :
regressor = LinearRegression()


#Train the model :
regressor.fit(X_train, y_train)

#get the coefficients :
L = regressor.coef_
print(L)


# organize the coefficient in a dataframe for better understanding
coeff_X = pd.DataFrame(regressor.coef_, index = dataset.columns[:-1], 
                       columns=['Coefficient'])


# Get the intercept of the our model
regressor.intercept_

#We can write the  Model's equation as following : 
print("Price = {:.2f} + {:.2f}*LotFrontage + {:.2f}*LotArea + {:.2f}*GarageCars + {:.2f}*GarageArea ".
      format(regressor.intercept_,L[0], L[1], L[2] ,L[3]))



# -----------------------------------------------------------------------------
#*****************  STEP 5 : Evaluate the Model - Predict( X_test) ************
# -----------------------------------------------------------------------------

# Predict the results for test set.
y_pred = regressor.predict(X_test)


#Compare between Actual output and the predicted values: 
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})




# Metrics and Model's evaluation :

from sklearn.metrics import mean_squared_error , r2_score

#Mean Squared Error:
mse = mean_squared_error(y_test, y_pred)

#Root Mean Squared Error:
root_mse = np.sqrt(mse)

#R_squared : 
R_squared = r2_score(y_test, y_pred)*100

print("Intercept:              ", regressor.intercept_)
print("Root Mean Square Error: ", root_mse)
print("R^2 Value in %:              ", R_squared)


#*******************STEP 6 : How to improve the Model ?

# Using more training data,
# Scaling and data standarization helps to avoid overfitting
# choosing more appropriate features which have high correlation to the output.


