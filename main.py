# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Dataset

ds = pd.read_csv('../input/mobile-price-classification/train.csv')

# Dividing Data into Features and Class as X and Y

X = ds.iloc[ : ,0:20].values
Y = ds.iloc[ : ,20].values

# Spliting Test Data and Train Data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=5)

# Scaling Test Data

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Appling All Algorithms

from sklearn.linear_model import LogisticRegression
LRc = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier
RFc = RandomForestClassifier(n_estimators=200)

from sklearn.tree import DecisionTreeClassifier
DTc = DecisionTreeClassifier()

from sklearn.svm import SVC
SVc = SVC()

# Training Test Data using Train Data

LRc.fit(X_train, Y_train)
RFc.fit(X_train, Y_train)
DTc.fit(X_train, Y_train)
SVc.fit(X_train, Y_train)

# Predicting Result

Y_pred_LR = LRc.predict(X_test)
Y_pred_RF = RFc.predict(X_test)
Y_pred_DT = DTc.predict(X_test)
Y_pred_SV = SVc.predict(X_test)

# Report Using Confusion Matrix

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(Y_test, Y_pred_LR)
cm_RF = confusion_matrix(Y_test, Y_pred_RF)
cm_DT = confusion_matrix(Y_test, Y_pred_DT)
cm_SV = confusion_matrix(Y_test, Y_pred_SV)

print("Confusion Matrix For Logistic Regression")
print(cm_LR)

print("Confusion Matrix For Decision Tree")
print(cm_DT)

print("Confusion Matrix For Random Forest")
print(cm_RF)

print("Confusion Matrix For Support Vector")
print(cm_SV)

# Accuracy Scores for Specific Algorithms

from sklearn.metrics import accuracy_score
ac_LR = accuracy_score(Y_test, Y_pred_LR)*100
ac_RF = accuracy_score(Y_test, Y_pred_RF)*100
ac_DT = accuracy_score(Y_test, Y_pred_DT)*100
ac_SV = accuracy_score(Y_test, Y_pred_SV)*100

print("Accuracy Score For Logistic Regression")
print(ac_LR, "%")

print("Accuracy Score For Decision Tree")
print(ac_DT, "%")

print("Accuracy Score For Random Forest")
print(ac_RF, "%")

print("Accuracy Score For Support Vector")
print(ac_SV, "%")

# End
