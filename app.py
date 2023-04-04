# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:07:08 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

#Loading the Saved Model

loaded_model = pickle.load(open('welltrained_model.sav', 'rb'))

#Creating a function for cost prediction

def cost_prediction(input_data) :
    new_input_data = input_data
    feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    input_data_with_names = dict(zip(feature_names, new_input_data))
    input_data_as_dataframe = pd.DataFrame([input_data_with_names])

    # Changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data_as_dataframe)

    # Reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print('$')
    return(prediction)


def main():
    #Giving a Title
    st.title('MEDICAL INSURANCE COST PREDICTION WEB APPLICATION')
    st.subheader('Created by Anuvab Chatterjee')
    
    #Getting Input Data From User
    age = st.text_input("Enter Your Age")
    sex = st.text_input("Male/Female[0/1]")
    bmi = st.text_input("Enter Your Body Mass Index")
    children = st.text_input("Enter Number of Children You Have")
    smoker = st.text_input("Smoker/Non-Smoker[0/1]")
    region = st.text_input("Enter Your Region[southeast-0/southwest-1/northeast-2/northwest-3]")
    
    #Code for Prediction
    cost = ''
    
    #Creating Button For Prediction
    if st.button('PREDICT COST (USD)') :
        cost = cost_prediction([age, sex, bmi, children, smoker, region])
        
    st.success(cost)



if __name__ == '__main__':
    main()
    
    
    
    