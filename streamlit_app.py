import streamlit as st
import runtime

#########################################################################################################################

#########################################################################################################################

st.set_page_config(page_title="Claim Prediction")         # webpage title
st.title("This page predicts if a Healthcare claim is fraudulent or not")



#https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader

#uploading files
uploaded_file1=st.file_uploader(label='Upload Beneficiary excel file',key=1)
uploaded_file2=st.file_uploader(label='Upload Inpatient excel file',key=2)
uploaded_file3=st.file_uploader(label='Upload Outpatient excel file',key=3)
uploaded_file4=st.file_uploader(label='Upload Provider excel file',key=4)


if ((uploaded_file1 is not None)&(uploaded_file2 is not None)&(uploaded_file3 is not None)&(uploaded_file4 is not None)):
    if(file_check(uploaded_file1,uploaded_file2,uploaded_file3,uploaded_file4)):
        st.write("Now , you are all set for prediction. Click Submit Button to get the results")
        uploaded_file1.seek(0)
        beneficiary = pd.read_csv(uploaded_file1)
        uploaded_file2.seek(0)
        inpatient=pd.read_csv(uploaded_file2)
        uploaded_file3.seek(0)
        outpatient=pd.read_csv(uploaded_file3)
        uploaded_file4.seek(0)
        provider=pd.read_csv(uploaded_file4)
        with st.form("Fraud Prediction",clear_on_submit=True):
            if st.form_submit_button("Submit"):
                predictions(beneficiary,inpatient,outpatient,provider)
    else:
        st.write('Upload correct file')

else:
    st.write('Upload all files')







