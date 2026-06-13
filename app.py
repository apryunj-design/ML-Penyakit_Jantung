# Library heart desease prediction
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image


# Configurate Streamlit app
st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Dashboard Machine Learning

created by : [Apriyanto](https://www.linkedin.com/in/apriyanto19/)
""")

add_selectitem = st.sidebar.selectbox("Tools:", ("Prediksi Penyakit Jantung"))

st.write("""
        # Aplikasi memprediksi **Penyakit Jantung**
        
        Dataset tes berasal dari [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)


# Collects user input features into dataframe
st.sidebar.header('INPUT DATA:')
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            cp = st.sidebar.selectbox('Tipe Nyeri Dada', [1, 2, 3, 4])
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan: ", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1 
            age = st.sidebar.slider("Usia", 20, 100, 50)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

#List of image file names
image_files = ['man-heart-attack.jpg', 'woman-heart-attack.jpg']

#Desired image size in pixels
desired_width = 160
desired_height = 160

#Display resized images using Streamlit's layout options
col1, col2, col3, col4 = st.columns(4)

for idx, image_file in enumerate(image_files):
        img = Image.open(image_file)
        
#Resized the image to the desired size
        resized_img = img.resize((desired_width, desired_height))

#Display resized images in respective columns
        if idx == 0:
                col1.image(resized_img, width="stretch")
        else:
                col2.image(resized_img, width="stretch")
    
#Loading images
heartdisease= Image.open('heart-disease.jpg')
strongheart =Image.open('strong-heart.jpg')

loaded_model = None
output = " "

if st.sidebar.button('Lakukan Prediksi'):
    df = input_df
    st.write(df)
    with open("best_model.pkl", 'rb') as file:  
        loaded_model = pickle.load(file)

if loaded_model is not None:        
        prediction = loaded_model.predict(df)
        if (prediction == 0).any():
            result = 'TIDAK ADA PENYAKIT JANTUNG'
        else:
            result = 'ADA POTENSI PENYAKIT JANTUNG'
        
        output = str(result)

        st.subheader('Hasil Prediksi: ')
        
        with st.spinner('Sedang memproses...'):
                time.sleep(4)
          
        st.success(f"{output}")
        
        if (prediction == 0).any():
                st.image(strongheart)
        else:
                st.image(heartdisease)