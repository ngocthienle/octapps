# -*- coding: utf-8 -*-
"""
Created on Friday, October 01 07:43:53 2021
This the third apps in the Stroke Project: Monintoring and tracking rehabilitation
of stroke patient.
@author: thienle
"""
import streamlit as st
from PIL import Image

def app():
    image = Image.open('eye_logo.png')
    st.image(image)

    st.title("""AI Healthcare Application for Ophthalmologic Diagnosis and Treatment in Thailand""")
    st.write(" ------ ")
    st.subheader("""Goals""")
    st.write("- The electronic patient information: Agerelated Macular Degeneration (AMD) and Diabetic Macular Edema (DME)")
    st.write("- The AI models that classifies AMD and DME from color fundus and OCT images")
    st.write("- The web-apps system of AMD and DME diagnosis for Ophthalmologist")
    
    expander_faq = st.expander("More About The Project")
    expander_faq.write("Thank you for visiting! If you have any questions about our project, please contact:")
    expander_faq.write("Professor Dr. Watit Benjapolakul: watit.b@chula.ac.th")

    
    
    
    
