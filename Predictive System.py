# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pickle

#Loading the Saved Model

loaded_model = pickle.load(open('C:/Users/Lenovo/OneDrive/Desktop/ML_PROJECT_MED_INSURANCE/trained_model.sav', 'rb'))

input_data = (30,0,22.85,1,0,3)
feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
input_data_with_names = dict(zip(feature_names, input_data))
input_data_as_dataframe = pd.DataFrame([input_data_with_names])

# Changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data_as_dataframe)

# Reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print('$',prediction)