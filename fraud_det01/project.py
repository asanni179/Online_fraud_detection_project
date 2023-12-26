# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:17:17 2023

@author: Windows
"""
  

import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

model = load('newonline_fraud.joblib')
navigation = st.sidebar.radio('Navigation',['Home','Contribution','Prediction']) 
def preprocess_categorical(df):
   lb = LabelEncoder()
   
   df['type'] = lb.fit_transform(df['type'])
   
   return df

def preprocess_numerical(df):
    # Scale numerical columns using StandardScaler
    scaler = MinMaxScaler()
    numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def preprocessor(input_df):
    # Preprocess categorical and numerical columns separately
    input_df = preprocess_categorical(input_df)
    input_df = preprocess_numerical(input_df)
    return input_df

def main():
    st.title('Online fraud detection app')
    st.write('This app is built to detect if there is going to be a fraud.')

    input_data = {}
    
    input_data['step'] = st.number_input('Steps', min_value = 0)
    input_data['type'] = st.selectbox('TYPE', ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT'])
    input_data['amount'] = st.number_input('amount')
    input_data['oldbalanceOrg'] = st.number_input('oldbalanceOrg')
    input_data['newbalanceOrig'] = st.number_input('newbalanceOrig')
    input_data['oldbalanceDest'] = st.number_input('oldbalanceDest')
    input_data['newbalanceDest'] = st.number_input('newbalanceDest')
    input_data['isFlaggedFraud'] = st.number_input('isFlaggedFraud', min_value = 0, max_value = 1)
    

    input_df = pd.DataFrame([input_data])
    st.write(input_df)

    if st.button('Predict'):
        final_df = preprocessor(input_df)
        prediction = model.predict(final_df)[0]
        
        if prediction == 1:
            st.write('There is a likelihood that there will be a fraud.')
        else:
            st.write('There is a likelihood that the transaction is a fraud')
if __name__ == '__main__':
        main()    