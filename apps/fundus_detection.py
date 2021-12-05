# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:54:08 2021

@author: thienle
"""
import cv2
import numpy as np
# Import libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input as v3_preprocess_input

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(
        "D:\OneDrive\streamlitprojects\eyes_web_apps\\aimodels\classification_AMD_Classification_InceptionV3.hdf5"
    )

    return model


with  st.spinner('Loading Model Into Memory...'):
    model = load_model()


def app():
    image = Image.open('D:\OneDrive\streamlitprojects\eyes_web_apps/eye_logo.png')
    st.image(image)
    st.title("""Fundus AMD Classification""")
    # Create header explaination
    my_expander = st.expander("See explanation", expanded=False)
    with my_expander:
        st.write("""
                 This web-apps examines a fundus image as Normal, Wet AMD, or Dry AMD using backend AI engine.
                 The trained AI engine is based on the InceptionV3 deep learning model.
                     """)
        image = Image.open('D:\OneDrive\streamlitprojects\eyes_web_apps\example_fundus.png')
        st.image(image,
                 caption='Example AMD classification.')

    st.subheader('Upload a fundus image and get the examination')
    uploaded_file = st.file_uploader("Upload fundus image")

    map_dict = {1: 'Dry AMD.',
                2: 'Normal.',  # hge
                0: 'Wet AMD.',  # hge_eyeball
                }
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB", use_column_width=False, width=300)
        # If the trained AI model using normalized divide (/255) images, then the images for predicting also divide 255.
        resized = v3_preprocess_input(resized)  # Normalized the input images by divide 255.
        img_reshape = resized[np.newaxis, ...]

        Genrate_pred = st.button("Examine")
        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            if prediction == 0: st.error("The retina is {}".format(map_dict[prediction]))
            if prediction == 1: st.error("The retina is {}".format(map_dict[prediction]))
            if prediction == 2: st.info("The retina is {}".format(map_dict[prediction]))