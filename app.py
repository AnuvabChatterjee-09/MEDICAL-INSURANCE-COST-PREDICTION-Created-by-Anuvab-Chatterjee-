# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:07:08 2023

@author: Lenovo
"""
import numpy as np
import pickle
import streamlit as st

#Loading the Saved Model

loaded_model = pickle.load(open('welltrained_model.sav', 'rb'))

#Creating a function for cost prediction

def cost_prediction(age, sex, bmi, children, smoker, region) :
    

    prediction = loaded_model.predict(np.array([[age, sex, bmi, children, smoker, region]]))
    print(prediction)
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
        cost = cost_prediction(age, sex, bmi, children, smoker, region)
        
    st.success(cost)



if __name__ == '__main__':
    main()
    
    
    
    