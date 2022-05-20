# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:35:28 2022

@author: Fatin
"""

import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

import streamlit as st

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('cpu')

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



# if ML, follow load model according to slide for ML

# DL

OHE_SCALER_PATH = os.path.join(os.getcwd(), 'ohe_scaler.pkl')
MMS_SCALER_PATH = os.path.join(os.getcwd(), 'mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')

ohe_scaler = pickle.load(open(OHE_SCALER_PATH, 'rb'))
mms_scaler = pickle.load(open(MMS_SCALER_PATH, 'rb'))

# ML
# model = pickle.load(open(PATH))

# DL
model = load_model(MODEL_PATH)
model.summary()

heart_chance = {0:'Negative', 1:'Positive'}

#%% Deployment

# use patient info

patient_info_full = [[65,1,3,142,220,1,0,158,0,2.3,1,0,1],
                     [61,1,0,140,207,0,0,138,1,1.9,2,1,3],
                     [45,0,1,128,204,0,0,172,0,1.4,2,0,2],
                     [40,0,1,125,307,0,1,162,0,0,2,0,2],
                     [48,1,2,132,254,0,1,180,0,0,2,0,2],
                     [41,1,0,108,165,0,0,115,1,2,1,0,3],
                     [36,0,2,121,214,0,1,168,0,0,2,0,2],
                     [45,1,0,111,198,0,0,176,0,0,2,1,2],
                     [57,1,0,155,271,0,0,112,1,0.8,2,0,3],
                     [69,1,2,179,273,1,0,151,1,1.6,1,0,3]]


patient_outcome = []
patient_chance = []

for i in range (len(patient_info_full)):
    
    patient_info_scaled = mms_scaler.transform(np.expand_dims(np.array(patient_info_full[i]), axis=0))
    outcome = model.predict(patient_info_scaled)
    patient_outcome.append(np.argmax(outcome))
    patient_chance.append(heart_chance[np.argmax(outcome)])


df_patient = pd.DataFrame(patient_info_full,
                          columns = ['age' , 'sex', 'cp' , 'trtbps', 'chol', 
                                     'fbs', 'restecg', 'thalachh', 'exng', 
                                     'oldpeak', 'slp', 'caa', 'thall'])

df_patient['outcome'] = patient_outcome
df_patient['chance'] = patient_chance


#%% Streamlit
with st.form('Heart Attack Prediction Form'):
    st.write("Patient's info")
    age = int(st.number_input('age'))
    sex = int(st.number_input('sex (0: male, 1: female)'))
    cp = int(st.number_input('cp'))
    trtbps = int(st.number_input('trtbps'))
    chol = int(st.number_input('chol'))
    fbs = int(st.number_input('fbs'))
    restecg = int(st.number_input('rest ecg'))
    thalachh = int(st.number_input('thalachh'))
    exng = int(st.number_input('exng'))
    oldpeak = st.number_input('old peak')
    slp = int(st.number_input('slp'))
    caa = int(st.number_input('caa'))
    thall = int(st.number_input('thall'))

    submitted = st.form_submit_button('Submit')
    if submitted == True:
        patient_info = np.array([age, sex, cp , trtbps, chol, 
                                 fbs, restecg, thalachh, exng, 
                                 oldpeak, slp, caa, thall])

        patient_info_scaled = mms_scaler.transform(np.expand_dims(patient_info, axis=0))
        
        outcome = model.predict(patient_info_scaled)
        
        if np.argmax(outcome)==1:
            st.warning("Risk of heart attack is positive. Please consult with doctor.")
            
        else:
            st.success('Risk of heart attack is negative. Keep maintaining a healthy lifestyle.')
        