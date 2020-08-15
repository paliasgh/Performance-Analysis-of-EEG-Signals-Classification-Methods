#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:33:34 2020

@author: paliasgh
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

df=pd.read_csv('training_matrix.csv')

x = df.drop(["Label"],axis=1).to_numpy()
y = df.loc[:,'Label'].values

y_1h = to_categorical(y)

# %% Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [5,10,50,100,200,500]:
    print(j)
    accr = []
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
           
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train,y_train)
            x_lda = lda.transform(x_train)
            x_lda_test = lda.transform(x_test)
            
            pca = decomposition.PCA(n_components=j)
            pca.fit(x_train)
            x_pca = pca.transform(x_train)
            x_pca_test = pca.transform(x_test)
            
            #clf = LogisticRegression(random_state=0,max_iter=1000).fit(x_train, y_train)
            clf = LogisticRegression(random_state=0,max_iter=1000).fit(x_lda, y_train)
            #clf = LogisticRegression(random_state=0,max_iter=5000).fit(x_pca, y_train)
           
            #preds = clf.predict(x_test)
            preds = clf.predict(x_lda_test)
            #preds = clf.predict(x_pca_test)
            
            cc = 0                                                                 
            for t in range(len(y_test)):      
                if preds[t] == y_test[t]:
                    cc += 1
                    
            accr.append(cc/len(y_test)*100)
            print(cc/len(y_test)*100)
             
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    
    
# %% SVM

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [0.1,1,10,50,100]:
    print(j)
    accr = []
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            
            #clf = SVC(C=j,kernel='poly',degree=2,gamma='auto').fit(x_train, y_train)
            clf = SVC(C=j,kernel='rbf',gamma='auto').fit(x_train, y_train)

            preds = clf.predict(x_test)

            cc = 0                                                                 
            for t in range(len(y_test)):      
                if preds[t] == y_test[t]:
                    cc += 1
                    
            accr.append(cc/len(y_test)*100)
            print(cc/len(y_test)*100)
             
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    
# %% Decision Tree

from sklearn.model_selection import KFold
from sklearn import tree

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [0.1]:
    print(j)
    accr = []
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            
            #clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=5).fit(x_train, y_train)
            clf = tree.DecisionTreeClassifier(criterion='gini',random_state=5).fit(x_train, y_train)

            preds = clf.predict(x_test)

            cc = 0                                                                 
            for t in range(len(y_test)):      
                if preds[t] == y_test[t]:
                    cc += 1
                    
            accr.append(cc/len(y_test)*100)
            print(cc/len(y_test)*100)
            
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    
# %% MLP
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [4,8,16,32,64,128,256,512,1024]:
    accr = []
    print(j)
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_1h[train_index], y_1h[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
                    
            model = Sequential()
            model.add(Flatten())
            model.add(Dense(j, activation='relu'))
            model.add(Dense(j, activation='relu'))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
            
            train = model.fit(x_train, y_train,    # Training
                          epochs=5, batch_size=1,
                          validation_data=(x_test, y_test),verbose=0)
            
            test = model.evaluate(x_test,y_test)   # Testing
            
            preds = model.predict(x_test)
            predict_class = np.argmax(preds, axis=1)
            true_class = np.argmax(y_test, axis=1)
                    
            accr.append(test[1]*100)
            if test[1] > accr_t:
                con = confusion_matrix(true_class, predict_class)
            print(test[1])
            accr_t = test[1]
    
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    

# %% CNN
accr_un = []
var_un = []

accr_t = 0
for j in [4,8,16,32,64]:
    accr = []
    print(j)
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_1h[train_index], y_1h[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
                    
            x_train_LSTM = np.reshape(x_train, (x_train.shape[0],1,x.shape[1]))
            x_test_LSTM = np.reshape(x_test, (x_test.shape[0],1,x.shape[1]))
            
            model = Sequential()
            model.add(Conv1D(j, 3, padding='same',activation='relu',input_shape=(1,x_train_LSTM.shape[2])))
            model.add(MaxPooling1D(2, padding='same'))
            model.add(Conv1D(j, 3, padding='same', activation='relu'))
            model.add(MaxPooling1D(2, padding='same'))
            model.add(Flatten())
            model.add(Dense(64, activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
            
            train = model.fit(x_train_LSTM, y_train,    # Training
                          epochs=5, batch_size=1,
                          validation_data=(x_test_LSTM, y_test),verbose=0)
            
            test = model.evaluate(x_test_LSTM,y_test)   # Testing
            
            preds = model.predict(x_test_LSTM)
            predict_class = np.argmax(preds, axis=1)
            true_class = np.argmax(y_test, axis=1)
                    
            accr.append(test[1]*100)
            if test[1] > accr_t:
                con = confusion_matrix(true_class, predict_class)
            print(test[1])
            accr_t = test[1]
    
    print(np.mean(accr),np.var(accr))
    accr_un.append(np.mean(accr))
    var_un.append(np.var(accr))


# %% LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [4,8,16,32,64,128,256,512,1024,2048]:
    accr = []
    print(j)
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_1h[train_index], y_1h[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
                    
            x_train_LSTM = np.reshape(x_train, (x_train.shape[0],1,x.shape[1]))
            x_test_LSTM = np.reshape(x_test, (x_test.shape[0],1,x.shape[1]))
                    
            model = Sequential()
            model.add(SimpleRNN(j, input_shape=(1,x_train_LSTM.shape[2]),activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
            
            train = model.fit(x_train_LSTM, y_train, 
                              epochs = 20, validation_data= (x_test_LSTM, y_test),verbose=0)
            
            test = model.evaluate(x_test_LSTM, y_test)   # Testing
            
            preds = model.predict(x_test_LSTM)
            predict_class = np.argmax(preds, axis=1)
            true_class = np.argmax(y_test, axis=1)
                    
            accr.append(test[1]*100)
            if test[1] > accr_t:
                con = confusion_matrix(true_class, predict_class)
            print(test[1])
            accr_t = test[1]
    
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))


# %% XGBoost

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

accr_hidd = []
var_hidd = []
accr_t = 0
for j in [0.9,1]:
    param = {         # The found optimal set of parameters:
    'max_depth': 2,   ## Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    'eta': j,         ## Step size shrinkage used in update to prevents overfitting.
    'subsample': 0.5, ## Subsample ratio of the training instances.
    'objective': 'multi:softmax',  # Error evaluation for multiclass training
    'num_class': 3}   # The number of classes that exist in this datset
    num_round = 100   # The number of training iterations            

    accr = []
    print(j)
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
                    
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            
            data_train = xgb.DMatrix(data=x_train,label=y_train)
            data_test = xgb.DMatrix(data=x_test,label=y_test)
            
            bst = xgb.train(param, data_train, num_round)
            preds = bst.predict(data_test)

                    
            predict_class = preds
            true_class = y_test
            
            cc = 0                                                                 
            for t in range(len(y_test)):      
                if preds[t] == y_test[t]:
                    cc += 1
            
            print(cc/len(y_test)*100)
            accr.append(cc/len(y_test)*100)
            if cc/len(y_test)*100 > accr_t:
                con = confusion_matrix(true_class, predict_class)
                accr_t = cc/len(y_test)*100
            
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    
# %% Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import tree

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [10,20,50,100]:
    print(j)
    accr = []
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
           
            clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1),
                                      learning_rate=1,n_estimators=j, random_state=0).fit(x_train, y_train)

            preds = clf.predict(x_test)

            cc = 0                                                                 
            for t in range(len(y_test)):      
                if preds[t] == y_test[t]:
                    cc += 1
                     
            accr.append(cc/len(y_test)*100)
            print(cc/len(y_test)*100)
             
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    
# %% Random Forest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import tree

accr_hidd = []
var_hidd = []

accr_t = 0
for j in [100,200,300,400]:
    print(j)
    accr = []
    for i in range(1):
        kf = KFold(n_splits=10,shuffle = True,random_state = i+2)
        for train_index, test_index in kf.split(x):     
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
           
            clf = RandomForestClassifier(n_estimators=j,criterion='gini',
                                         max_depth=10, random_state=0).fit(x_train, y_train)

            preds = clf.predict(x_test)

            cc = 0                                                                 
            for t in range(len(y_test)):      
                if preds[t] == y_test[t]:
                    cc += 1
                     
            accr.append(cc/len(y_test)*100)
            print(cc/len(y_test)*100)
             
    print(np.mean(accr),np.var(accr))
    accr_hidd.append(np.mean(accr))
    var_hidd.append(np.var(accr))
    