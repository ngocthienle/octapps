# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:57:43 2021
OCT Eyes Image Classification project - Chula
Three classes:
 - Normal (Healthy)
 - AMD (age-related macular degeneration)
 - Others (Diabetic Retinopathy, Macular Hole, etc)

@author: thienle
Date: July 14, 2021
"""
# Import libraries
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# NAME OF PAGE
logo_image = Image.open('logo.png')
st.set_page_config(page_title = 'OCT Diseases Detection', page_icon=logo_image)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("\oct3\OCT3_EyeDiseases_Classification_MobileNetV2.hdf5")
    return model

with st.spinner('Loading Model Into Memory...'):
    model = load_model()

# Create web-app title
st.title("""OCT Eyes diseases detection Web Application Chulalongkorn University""")

# Create header explaination
st.write("""
    This web-apps predicts the input OCT image is normal, AMD, or other diseases.
    The backend system is a trained AI model. The detail diseases in each catelogy as below:
        
        1. normal: healthy eyes.
        2. AMD: age related macular degeneration (Drusen, Choroidal neovascularization)
        3. orther diseases: diabetic macular edema, Central-Serous-Retionpathy, Diabetic Retinopathy
    """)
    
st.subheader('Example input OCT eye images')
image = Image.open('example_pics.jpg')
st.image(image, caption='Example of input OCT images.')

st.subheader('Choose a OCT image and get the output prediction')
uploaded_file = st.file_uploader("Upload your input jpeg file")

map_dict = {0: 'normal.',
            1: 'AMD', # hge
            2: 'other diseases.', #hge_eyeball
            }

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("The OCT image is {}".format(map_dict [prediction]))
