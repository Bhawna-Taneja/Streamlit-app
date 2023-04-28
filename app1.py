import streamlit as st
import pickle
import numpy as np
import sklearn
import webbrowser

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

if st.button('Home'):
    # add a link to the button
    href = 'https://website-prediction.s3.jp-tok.cloud-object-storage.appdomain.cloud/index.html'
    target = '_blank'
    st.markdown(f'<a href="{href}" target="{target}">Home</a>', unsafe_allow_html=True)
#if st.button('Home'):
    #webbrowser.open_new_tab('https://website-prediction.s3.jp-tok.cloud-object-storage.appdomain.cloud/index.html')
    #st.markdown(webbrowser.open_new_tab('https://website-prediction.s3.jp-tok.cloud-object-storage.appdomain.cloud/index.html'))
st.title("Laptop Predictor")

# brand
Company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
TypeName = st.selectbox('Type',df['TypeName'].unique())

# Ram
Ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
Weight = st.number_input('Weight of the Laptop')

# Touchscreen
TouchScreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
Ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
                                               '2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
Cpu_brand = st.selectbox('CPU',df['Cpu_brand'].unique())

HDD = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

SSD = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

Gpu_brand = st.selectbox('GPU',df['Gpu_brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = 0
    if TouchScreen == 'Yes':
        TouchScreen = 1
    else:
        TouchScreen = 0

    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
   # print(X_res)
   # print(Y_res)
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([Company,TypeName,Ram,Weight,TouchScreen,Ips,ppi,Cpu_brand,HDD,SSD,Gpu_brand ,os],dtype = object)
    query = query.reshape(1,12)
    st.title("The predicted price is " + str(int(np.exp(pipe.predict(query)))))
