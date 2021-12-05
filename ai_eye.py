# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:32:36 2021
The multi-page web application for stroke project- Chula
@author: Ngoc Thien Le, Postdoc Researcher
"""
import streamlit as st
from PIL import Image

logo_image = Image.open('D:\OneDrive\streamlitprojects\eyes_web_apps/logo.png')

st.set_page_config(layout="centered", page_title='AI Eyecare',
                   page_icon=logo_image)

from multiapp import MultiApp
from apps import patient, introduction, oct_detection, fundus_detection  # Import the app modules want here.

app = MultiApp()

# Add all your application here
app.add_app("Introduction", introduction.app)
app.add_app("Patient Record", patient.app)
app.add_app("Fundus Retina Image", fundus_detection.app)
app.add_app("OCT Image", oct_detection.app)

# The main apps
app.run()
