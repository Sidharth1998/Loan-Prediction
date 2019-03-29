#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:40:10 2018

@author: sidharthdugar
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#getting dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
columns = dict(test_data['Loan_ID'])
train_data = train_data.drop(['Loan_ID'],axis=1)
test_data = test_data.drop(['Loan_ID'],axis=1)


#TRAIN DATA
#handling null or nan values in train_data
#->Removing nan rows in married column because there were no nan in test_data and 
  #the number of nan rows were less in train_data
train_data = train_data[pd.notnull(train_data['Married'])]
#->taking care of null values in gender
train_data['Gender'] = train_data['Gender'].fillna(value='Male')
#->Dependents
train_data['Dependents'] = train_data['Dependents'].fillna(value='0')
train_data['Self_Employed'] = train_data['Self_Employed'].fillna(value='No')
train_data['LoanAmount'] = train_data['LoanAmount'].fillna(value=round(train_data['LoanAmount'].mean()))
#for Loan_Amount_Term i took the mode value
train_data['Loan_Amount_Term'] = train_data['Loan_Amount_Term'].fillna(value=360)
train_data['Credit_History'] = train_data['Credit_History'].fillna(value=1)

#TEST DATA
#handling null or nan values in test_data
#->Removing nan rows in married column because there were no nan in test_data and 
  #the number of nan rows were less in train_data
test_data = test_data[pd.notnull(test_data['Married'])]
#->taking care of null values in gender
test_data['Gender'] = test_data['Gender'].fillna(value='Male')
#->Dependents
test_data['Dependents'] = test_data['Dependents'].fillna(value='0')
test_data['Self_Employed'] = test_data['Self_Employed'].fillna(value='No')
test_data['LoanAmount'] = test_data['LoanAmount'].fillna(value=round(train_data['LoanAmount'].mean()))
#for Loan_Amount_Term i took the mode value
test_data['Loan_Amount_Term'] = test_data['Loan_Amount_Term'].fillna(value=360)
test_data['Credit_History'] = test_data['Credit_History'].fillna(value=1)

#Encoding categorical data
train_data = pd.get_dummies(train_data,drop_first=True)
test_data = pd.get_dummies(test_data,drop_first=True)


matrix = train_data.corr()
#train_data = train_data.drop(['Loan_Amount_Term','CoapplicantIncome','Education_Not Graduate'],axis=1)
#test_data = test_data.drop(['Loan_Amount_Term','CoapplicantIncome','Education_Not Graduate'],axis=1)

#Separating 
X_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values
X_test = test_data.iloc[:,:].values

#Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#ML Model
from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#K-Cross Validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(classifier,X_train,y_train,n_jobs = -1,cv=10)
accuracies.mean()

#Storing the predictions into a csv file
pid=[]
for values in columns.values():
    pid.append(values)
y_pred = pd.DataFrame(data=y_pred)
predictions = pd.DataFrame()
predictions.insert(loc=0,column="Loan_ID",value=pid)
predictions.insert(loc=1,column="Loan_Status",value=y_pred)
predictions.to_csv('predictions.csv',index=False)