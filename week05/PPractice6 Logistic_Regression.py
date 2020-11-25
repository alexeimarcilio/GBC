# Info:A Logistic Regression Code to diagnose diabetes

# Author: M. Fadaee
# Date: Feb 2020

# Purpose: diagnose diabetes, using Logistic Reg.

# inputs: Medical data for ~800 samples

# outputs: Probability of a person having diabetes

# Version control: V1

#------------------------------------------------------------------------------

# Pandas Dataframes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-- Read the dataset
data = pd.read_csv ("diabetes_data.csv")

#-- get column names
col_names = data.columns

#-- get the featuers and the target values
x = data.iloc[:, 0:7]
y = data.iloc[:, 8]

#-- get the number of samples and featuers
samples = x.shape[0]
featuers = x.shape[1]

#-- Split data into Train and Test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2)

#-- Built the Logostic Regression Model----------------------------------------

#-- define the model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

#-- Fit the model
log_reg.fit(x_train, y_train)

#-- predict for the test set
y_pred = log_reg.predict (x_test)

y_probabilities = log_reg.predict_proba(x_test)

#-- find the confusion matrix
from sklearn import metrics
confusion_mat = metrics.confusion_matrix(y_test, y_pred)

#-- Print the Conf. Mat.
confusion_mat

#-- Plot the Conf. Mat. as a heatmap
import seaborn as sns

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(confusion_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix  !!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')




