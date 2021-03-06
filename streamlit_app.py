import streamlit as st
import pandas as pd
from pycaret.classification import *
import os
import requests
from prediction_module import file_check,preprocessing,feature_engineering,predictions


st.set_page_config(page_title="Claim Prediction")                                         # webpage title
st.title("This page predicts if a Healthcare claim is fraudulent or not")                 #Prints page Header.

#uploading files
uploaded_file1=st.file_uploader(label='Upload Beneficiary excel file',key=1)               
uploaded_file2=st.file_uploader(label='Upload Inpatient excel file',key=2)
uploaded_file3=st.file_uploader(label='Upload Outpatient excel file',key=3)
uploaded_file4=st.file_uploader(label='Upload Provider excel file',key=4)


if ((uploaded_file1 is not None)&(uploaded_file2 is not None)&(uploaded_file3 is not None)&(uploaded_file4 is not None)):     #when all files are uploaded, it goes inside
    if(file_check(uploaded_file1,uploaded_file2,uploaded_file3,uploaded_file4)):
        st.write("Now , you are all set for prediction. Click Submit Button to get the results")
        uploaded_file1.seek(0)
        beneficiary = pd.read_csv(uploaded_file1)                      #creates a dataframe for beneficiary data
        uploaded_file2.seek(0)
        inpatient=pd.read_csv(uploaded_file2)                          #creates a dataframe for inpatient data
        uploaded_file3.seek(0)
        outpatient=pd.read_csv(uploaded_file3)                         #creates a dataframe for outpatient data
        uploaded_file4.seek(0)                        
        provider=pd.read_csv(uploaded_file4)                           #creates dataframe for providers
        with st.form("Fraud Prediction",clear_on_submit=True):         #it creates a Submit form section with Submit button inside.
            if st.form_submit_button("Submit"):                        #when Submit button is clicked, it calls the predictions() function from prediction_module.
                results=predictions(beneficiary,inpatient,outpatient,provider)
                st.write(results)
    else:
        st.write('Upload correct file')    #when the file format is not correct, it shows below message.

else:                                      #when all files are not uploaded, it throws below message.
    st.write('Upload all files')







