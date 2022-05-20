# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:36:50 2022

@author: Fatin
"""

import pandas as pd
import os
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import numpy as np

DATA_PATH = os.path.join(os.getcwd(), 'heart.csv')

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

#%% EDA

# 1: Data Loading
df = pd.read_csv(DATA_PATH)

# 2: Data Inspection
df.info()
df.describe()

print(df.nunique())
print(df.isna().sum())

# 3: Data Cleaning

bool_series = pd.DataFrame(df).duplicated()
sum(bool_series==True)

print (len(df)) # Number of rows before removing duplicates
df = pd.DataFrame(df).drop_duplicates()

print (len(df)) # Number of rows after removing duplicates
pd.DataFrame(df).describe().T
pd.DataFrame(df).boxplot()

# 4: Feature Selection

X = df.drop(['output'], axis = 1)
y = df['output']

lasso = Lasso()
lasso.fit(X,y)
lasso_coef = lasso.coef_ # to obtain coefficients
print(lasso_coef) # select non-zero coefficiens

# graphs
plt.figure(figsize=(10,10))
plt.plot(X.columns,abs(lasso_coef))
plt.grid()

# 5: Data Preprocessing

X = df.iloc[:,0:13]
y = df.iloc[:,13]

mms_scaler = MinMaxScaler()
X_scaled = mms_scaler.fit_transform(X)
pickle.dump(mms_scaler, open('mms_scaler.pkl','wb'))

ohe_scaler = OneHotEncoder(sparse=False)
y_one_hot = ohe_scaler.fit_transform(np.expand_dims(y, axis=-1))
pickle.dump(ohe_scaler, open('ohe_scaler.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, 
                                                   test_size=0.3, random_state=123) 
#%% Model Creation

model = Sequential()
model.add(Dense(128, activation = ('relu'), input_shape=(13,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics='acc')

#%% Model Deployment

hist = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save(MODEL_SAVE_PATH)

#%% Model Evaluation

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
