import streamlit as st
import pickle
import numpy  as np
import pandas as pd
import webbrowser

# import the model
pipe  = pickle.load(open('LinearRegressionModek.pkl','rb'))
df = pickle.load(open('df1.pkl','rb'))

if st.button('Home'):
    webbrowser.open_new_tab('https://website-prediction.s3.jp-tok.cloud-object-storage.appdomain.cloud/index.html')
    
st.title("Car Predictor")

# company
company = st.selectbox('Company', df['company'].unique())

# model
model = st.selectbox('Model', df['name'].unique())

# Year of Purchase
year = st.selectbox('Year of Purchase', df['year'].unique())

# Fuel Type
fuel = st.selectbox('Type of Fuel', ['Diesel', 'Petrol'])

# Kms Driven
kms = st.number_input('Kms Driven')

if st.button('Predict Price'):
    # query
    query = [model,company,year,kms,fuel]
    data = [query]
    df = pd.DataFrame(data,columns=['name','company','year','kms_driven','fuel_type'])
    #query = query.reshape(1,5)

    #st.title("Predicted Price " + str(int(np.exp(pipe.predict(df)[0]))))
    st.title("Predicted Price " + str(int(pipe.predict(df))))
