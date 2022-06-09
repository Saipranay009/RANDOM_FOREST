# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:28:23 2022

@author: Sai pranay
"""

import pandas as pd
cd = pd.read_csv("E:\\DATA_SCIENCE_ASS\\RANDOM FOREST\\Company_Data (1).csv")
print(cd)
list(cd)
cd.shape
cd.info()
cd.value_counts()
cd.describe()
cd.dtypes

#------------------label_encoding_for_the_categorical_data---------------------

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
cd["ShelveLoc_"] = LE.fit_transform(cd["ShelveLoc"])
cd[["ShelveLoc", "ShelveLoc_"]].head(11)
cd.shape
pd.crosstab(cd.ShelveLoc,cd.ShelveLoc_)

cd["Urban_"] = LE.fit_transform(cd["Urban"])
cd[["Urban", "Urban_"]].head(11)
cd.shape
pd.crosstab(cd.Urban,cd.Urban_)

cd["US_"] = LE.fit_transform(cd["US"])
cd[["US", "US_"]].head(11)
cd.shape
pd.crosstab(cd.US,cd.US_)

list(cd)
cd.shape
#------------after_label_encoding_the_categorical_data-------------------------

cd1 = cd.iloc[:,11:]
print(cd1)
cd1.shape
list(cd1)

#-----------------------------------remaining_data-----------------------------

cd2 = cd.drop(['ShelveLoc','Urban','US','ShelveLoc_','Urban_','US_'],axis = 1)
print(cd2)
list(cd2)


#----------------------standardization_the_remaining_data----------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(cd2)
X_scale


X_new1 = pd.DataFrame(X_scale)
X_new1

cd_new = pd.concat([X_new1, cd1],axis = 1)
print(cd_new)
list(cd_new)
cd_new.shape
cd_new.describe()
cd_new.info()



x = cd_new.iloc[:,1:11]
print(x)
x.ndim

y = cd_new.iloc[:,0]
print(y)
y.ndim
y.shape

import numpy as np


Y1=[]
for i in range(0,400,1):
    if y.iloc[i,]>=y.mean():
        print('High')
        Y1.append('High')
    else:
        print('Low')
        Y1.append('Low')

Y_new=pd.DataFrame(Y1)
Y_new


#--------------------Splitting train and test data sets------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y_new, test_size=0.20,stratify=Y_new,random_state=78)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


#----------------------------------- RANDOM FORESTS----------------------------

from sklearn.ensemble import RandomForestClassifier # Classifier
RF = RandomForestClassifier(max_features = 0.3,n_estimators = 600, random_state = 88) 
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




