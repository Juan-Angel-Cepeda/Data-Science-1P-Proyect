import streamlit as st
import pandas as pd
import numpy as np
import prediction
import joblib
from combined_attributes_adder import CombinedAttributesAdder

st.header('House prediction base in Californa Prices Values DataSet')
st.write('Data Science course for predictions house value data')
col1, col2 = st.columns(2)

longitude = col1.number_input('Longitude', min_value = -124.0, max_value = -110.0, format = "%.2f")
total_rooms = col1.number_input('Total rooms', min_value = 1.0, max_value = 50000.0, format = "%.0f")
median_income = col1.number_input('Median income', min_value = 0.0, max_value = 17.0, format = "%.4f")
households = col1.number_input('Households', min_value = 1.0, max_value = 10000.0, format = "%.0f")
housing_median_age = col1.slider('Housing median age', step=1.0, min_value=1.0, max_value=100.0, value=0.0, format = "%.0f")

latitude = col2.number_input('Latitude', min_value = 30.0, max_value = 50.0, format = "%.2f")
total_bedrooms = col2.number_input('Total bedrooms', min_value = 1.0, max_value = 7000.0, format = "%.0f")
population = col2.number_input('Population', min_value = 1.0, max_value = 50000.0, format = "%.0f")
ocean_proximity = col2.selectbox('Ocean proximity', ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])

model = col2.radio(
    'Select the model to use:',
    (
        'Linear Regression', 
        'Decision Tree Regression', 
        'Random Forest Regression', 
        'Support Vector Regression'
    )
)

if st.button('Predict'):
    data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]}
    )

    result = prediction.predict(data, model)
    st.write("Valor predecido por el modelo ${} ${:.3f}".format(model,result[0]))

