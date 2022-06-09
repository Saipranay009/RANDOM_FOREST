# -*- coding: utf-8 -*-
"""
Created on Thu May 12 23:43:54 2022

@author: Sai pranay
"""
#--------------------------Importinf the data set------------------------------

import pandas as pd
fck = pd.read_csv("E:\\DATA_SCIENCE_ASS\\RANDOM FOREST\\Fraud_check.csv")
print(fck)
list(fck)
fck.shape
fck.info()
fck.value_counts()
fck.describe()
fck.dtypes

#------------------label_encoding_for_the_categorical_data---------------------

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
fck["Undergrad_"] = LE.fit_transform(fck["Undergrad"])
fck[["Undergrad", "Undergrad_"]].head(10)
fck.shape
pd.crosstab(fck.Undergrad,fck.Undergrad_)

fck["MaritalStatus_"] = LE.fit_transform(fck["Marital.Status"])
fck[["Marital.Status", "MaritalStatus_"]].head(10)
fck.shape

fck["Urban_"] = LE.fit_transform(fck["Urban"])
fck[["Urban", "Urban_"]].head(10)
fck.shape
pd.crosstab(fck.Urban,fck.Urban_)

list(fck)
fck.shape

#-----------------------------------Droping-----------------------------

fck1 = fck.drop(['Undergrad','Marital.Status','Urban'],axis = 1)
print(fck1)
list(fck1)


#------------after_label_encoding_the_categorical_data-------------------------

x = fck1.iloc[:,1:6]
print(x)
x.shape
list(x)


X_ = pd.DataFrame(x)
X_


y = fck1['Taxable.Income']
print(y)
y.ndim
y.shape

import numpy as np

Y1=[]
for i in range(0,600,1):
    if y.iloc[i,]<=30000:
        print('Risky')
        Y1.append('Risky')
    else:
        print('Good')
        Y1.append('Good')


Y_new=pd.DataFrame(Y1)
Y_new


#--------------------Splitting train and test data sets------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_new, test_size=0.25,stratify=Y_new,random_state=24)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


#----------------------------------- RANDOM FORESTS----------------------------

from sklearn.ensemble import RandomForestClassifier # Classifier
RF = RandomForestClassifier(max_features = 0.3,n_estimators = 400, random_state = 71) 
RF.fit(X_train,Y_train)
Y_pred = RF.predict(X_test)

Y_pred_1=pd.DataFrame(Y_pred)
Y_pred_1


from sklearn import metrics 


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,Y_pred)
acc = accuracy_score(Y_test,Y_pred).round(2)
acc
print(" accuracy score:" , acc)

###############################################################################
# Create two lists for training and test errors
training_Error = []
test_Error = []

# Define a range of 1 to 10 (included) neighbors to be tested
settings = np.arange(0.1, 1.1, 0.1)

# Loop with the  RandomForestClassifier through the Max depth values to determine the most appropriate (best)
from sklearn.ensemble import RandomForestClassifier # Classifier


for samp_val in settings:
    Classifier = RandomForestClassifier(n_estimators=500,random_state=42,max_features=samp_val)
    Classifier.fit(X_train, Y_train)

    Y_Train_pred = Classifier.predict(X_train)
    training_Error.append(np.sqrt(metrics.accuracy_score(Y_Train_pred, Y_train).round(3)))

    Y_Test_pred = Classifier.predict(X_test)
    test_Error.append(np.sqrt(metrics.accuracy_score(Y_Test_pred, Y_test).round(3)))

print(training_Error)
print(test_Error)


# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)

import matplotlib.pyplot as plt

plt.plot(settings, training_Error, label='RMSE of the training set')
plt.plot(settings, test_Error, label='RMSE of the test set')
plt.ylabel('Root accuracy score:')
plt.xlabel('Percentage of features in RF')
plt.legend()




