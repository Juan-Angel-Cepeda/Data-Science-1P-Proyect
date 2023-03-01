import streamlit as st
import pandas as pd
import numpy as np
import prediction
import joblib
from combined_attributes_adder import CombinedAttributesAdder


st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",width =50,use_column_width=3)
st.header('House prediction base in Californa Prices Values DataSet')
st.write('Data Science course for predictions house value data')

col1, col2, col3 = st.columns(3)

with st.container():
    st.write("Información Geográfica")
    longitude = col1.number_input('Longitud', min_value = -124.0, max_value = -110.0, format = "%.2f")
    latitude = col1.number_input('Latitud', min_value = 30.0, max_value = 50.0, format = "%.2f")
    population = col1.number_input('Población', min_value = 1.0, max_value = 50000.0, format = "%.0f")
    ocean_proximity = col1.selectbox('Proximidad al Oceano', ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])


with st.container():
    total_rooms = col2.number_input('Total de habitaciones', min_value = 1.0, max_value = 50000.0, format = "%.0f")
    total_bedrooms = col2.number_input('Total de dormitorios', min_value = 1.0, max_value = 7000.0, format = "%.0f")

with st.container():
    households = col3.number_input('Hogares / Households', min_value = 1.0, max_value = 10000.0, format = "%.0f")
    housing_median_age = col3.slider('Promedio de edad de vivienda / Housing median age', step=1.0, min_value=1.0, max_value=100.0, value=0.0, format = "%.0f")
    median_income = col3.number_input('Ingreso Promedio /Median income', min_value = 0.0, max_value = 17.0, format = "%.4f")

with st.container():
    model = col3.radio(
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
        st.write("Valor predecido por el modelo {} es de {:.1f} dlls".format(model,result[0]))

