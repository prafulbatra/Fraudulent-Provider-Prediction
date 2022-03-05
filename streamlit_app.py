import streamlit as st
import pandas as pd
from pycaret.classification import *
import os
import requests

##########################################################################################################
def file_check(uploaded_file1,uploaded_file2,uploaded_file3,uploaded_file4):
    beneficiary = pd.read_csv(uploaded_file1)
    inpatient=pd.read_csv(uploaded_file2)
    outpatient=pd.read_csv(uploaded_file3)
    provider=pd.read_csv(uploaded_file4)
    flag1=flag2=flag3=flag4=False
    if ((len(beneficiary.columns) ==25) and (beneficiary.columns[0] == 'BeneID')):
        flag1=True
    if ((len(inpatient.columns) ==30) and (inpatient.columns[1] == 'ClaimID')):
        flag2=True
    if ((len(outpatient.columns) ==27) and (outpatient.columns[1] == 'ClaimID')):
        flag3=True
    if ((len(provider.columns) ==1) and (provider.columns[0] == 'Provider')):
        flag4=True
    
    return (flag1&flag2&flag3&flag4)
###########################################################################################################
def preprocessing(test):
    #Preprocessing
    #1)converting all date columns from string to datetime datatype
    date_cols=["ClaimStartDt","ClaimEndDt","AdmissionDt","DischargeDt","DOB","DOD"] 
    for column in date_cols:
        test[column]=pd.to_datetime(test[column])

    #2)including both admission day and discharge day as hospitalization for inpatients
    test.loc[test["AdmissionDt"].isna()==False,"noDaysAdmit"]= (test["DischargeDt"] - test["AdmissionDt"]).dt.days + 1  
    test.loc[test["AdmissionDt"].isna()==True,"noDaysAdmit"]=0     #outpatients will have NaN and so making noDaysAdmit 0 for them

    test["noDaysClaim"]=(test["ClaimEndDt"] - test["ClaimStartDt"]).dt.days + 1
    #there are no null values in ClaimStartDt and ClaimEndDt


    #3)Adding new column age(if Date of Death is not given , age is calculated by subtracting DOB from Claim Start Date)
    test.loc[test["DOD"].isna()== False , "age"]= round((test['DOD']-test['DOB']).dt.days/365)
    test.loc[test["DOD"].isna()== True , "age"]= round((test['ClaimStartDt']-test['DOB']).dt.days/365)

    #4)Adding new column 'inOut', whether patient is inpatient or outpatient..0-means outpatient , 1 means inpatient
    test.loc[test["AdmissionDt"].isna()==True,"inOut"]=0
    test.loc[test["AdmissionDt"].isna()==False,"inOut"]=1

    #5)Adding column 'whetherAlive' whether patient is alive or dead..alive= 1 , dead =0
    test.loc[test["DOD"].isna()== True , "whetherAlive"]=1
    test.loc[test["DOD"].isna()== False , "whetherAlive"]=0

    #6)Replacing 2 with 0 in the below columns..0 means patient doesn't have that chronic disease and 1 if patient has that disease
    chronic_columns=["ChronicCond_Alzheimer", "ChronicCond_Heartfailure","ChronicCond_KidneyDisease","ChronicCond_Cancer","ChronicCond_ObstrPulmonary",
                     "ChronicCond_Depression","ChronicCond_Diabetes","ChronicCond_IschemicHeart","ChronicCond_Osteoporasis","ChronicCond_rheumatoidarthritis",
                     "ChronicCond_stroke","Gender"]
    for column in chronic_columns:
        test[column] = test[column].apply(lambda x: 0 if x == 2 else 1)    #there are no null values in any of these columns

    #7)Adding new column which is difference b/w claim and hospitalization duration
    test['ClaimMinusAdmitDays']=test['noDaysClaim']-test['noDaysAdmit']

    #8)Replacing 'Y' with 1 in RenalDiseaseIndicator
    test['RenalDiseaseIndicator'] = test['RenalDiseaseIndicator'].apply(lambda x: 0 if x == '0' else 1)

    #9)adding column 'PatientRiskValue' that sums all these existing columns and later we can drop them.
    chronic_columns=["ChronicCond_Alzheimer", "ChronicCond_Heartfailure","ChronicCond_KidneyDisease","ChronicCond_Cancer","ChronicCond_ObstrPulmonary",
                     "ChronicCond_Depression","ChronicCond_Diabetes","ChronicCond_IschemicHeart","ChronicCond_Osteoporasis","ChronicCond_rheumatoidarthritis",
                     "ChronicCond_stroke","RenalDiseaseIndicator"]

    test['PatientRiskValue']=0             #initialising the value to zero at first and later add values of all the columns
    for column in chronic_columns:
        test['PatientRiskValue']+=test[column]

    #10)Adding 2 new columns , TotalAnnualReimbursementAmt and TotalAnnualDeductibleAmt
    test['TotalAnnualReimbursementAmt']=test['IPAnnualReimbursementAmt'] + test['OPAnnualReimbursementAmt']
    test['TotalAnnualDeductibleAmt']=test['IPAnnualDeductibleAmt'] + test['OPAnnualDeductibleAmt']

    #11)Filling all NA values wih 0
    test=test.fillna(0).copy()

    return test
#########################################################################################################################

def feature_engineering(test):
    #1)Creating 15 new columns 'Mean_InscClaimAmtReimbursed_PerColumn' for physicians, beneficiary, claim diagnosis and procedure codes.
    test['Mean_InscClaimAmtReimbursed_PerAttendingPhysician']=test.groupby('AttendingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerOperatingPhysician']=test.groupby('OperatingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerOtherPhysician']=test.groupby('OtherPhysician')['InscClaimAmtReimbursed'].transform('mean')

    test['Mean_InscClaimAmtReimbursed_PerClmAdmitDiagnosisCode']=test.groupby('ClmAdmitDiagnosisCode')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerDiagnosisGroupCode']=test.groupby('DiagnosisGroupCode')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmDiagnosisCode_1']=test.groupby('ClmDiagnosisCode_1')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmDiagnosisCode_2']=test.groupby('ClmDiagnosisCode_2')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmDiagnosisCode_3']=test.groupby('ClmDiagnosisCode_3')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmDiagnosisCode_4']=test.groupby('ClmDiagnosisCode_4')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmDiagnosisCode_5']=test.groupby('ClmDiagnosisCode_5')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmDiagnosisCode_6']=test.groupby('ClmDiagnosisCode_6')['InscClaimAmtReimbursed'].transform('mean')

    test['Mean_InscClaimAmtReimbursed_PerClmProcedureCode_1']=test.groupby('ClmProcedureCode_1')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmProcedureCode_2']=test.groupby('ClmProcedureCode_2')['InscClaimAmtReimbursed'].transform('mean')
    test['Mean_InscClaimAmtReimbursed_PerClmProcedureCode_3']=test.groupby('ClmProcedureCode_3')['InscClaimAmtReimbursed'].transform('mean')

    test['Mean_InscClaimAmtReimbursedPerBeneID']=test.groupby('BeneID')['InscClaimAmtReimbursed'].transform('mean')



    #2)Creating 15 new columns 'Mean_DeductibleAmtPaid_PerColumn' for physicians, beneficiary, claim diagnosis and procedure codes.

    test['Mean_DeductibleAmtPaid_PerAttendingPhysician']=test.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerOperatingPhysician']=test.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerOtherPhysician']=test.groupby('OtherPhysician')['DeductibleAmtPaid'].transform('mean')

    test['Mean_DeductibleAmtPaid_PerClmAdmitDiagnosisCode']=test.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerDiagnosisGroupCode']=test.groupby('DiagnosisGroupCode')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmDiagnosisCode_1']=test.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmDiagnosisCode_2']=test.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmDiagnosisCode_3']=test.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmDiagnosisCode_4']=test.groupby('ClmDiagnosisCode_4')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmDiagnosisCode_5']=test.groupby('ClmDiagnosisCode_5')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmDiagnosisCode_6']=test.groupby('ClmDiagnosisCode_6')['DeductibleAmtPaid'].transform('mean')

    test['Mean_DeductibleAmtPaid_PerClmProcedureCode_1']=test.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmProcedureCode_2']=test.groupby('ClmProcedureCode_2')['DeductibleAmtPaid'].transform('mean')
    test['Mean_DeductibleAmtPaid_PerClmProcedureCode_3']=test.groupby('ClmProcedureCode_3')['DeductibleAmtPaid'].transform('mean')

    test['Mean_DeductibleAmtPaidPerBeneID']=test.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')


    #3)Creating 15 new columns 'Mean_IPAnnualReimbursementAmt_PerColumn' for physicians, beneficiary, claim diagnosis and procedure codes.
    test['Mean_IPAnnualReimbursementAmt_PerAttendingPhysician']=test.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerOperatingPhysician']=test.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerOtherPhysician']=test.groupby('OtherPhysician')['IPAnnualReimbursementAmt'].transform('mean')

    test['Mean_IPAnnualReimbursementAmt_PerClmAdmitDiagnosisCode']=test.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerDiagnosisGroupCode']=test.groupby('DiagnosisGroupCode')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmDiagnosisCode_1']=test.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmDiagnosisCode_2']=test.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmDiagnosisCode_3']=test.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmDiagnosisCode_4']=test.groupby('ClmDiagnosisCode_4')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmDiagnosisCode_5']=test.groupby('ClmDiagnosisCode_5')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmDiagnosisCode_6']=test.groupby('ClmDiagnosisCode_6')['IPAnnualReimbursementAmt'].transform('mean')

    test['Mean_IPAnnualReimbursementAmt_PerClmProcedureCode_1']=test.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmProcedureCode_2']=test.groupby('ClmProcedureCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    test['Mean_IPAnnualReimbursementAmt_PerClmProcedureCode_3']=test.groupby('ClmProcedureCode_3')['IPAnnualReimbursementAmt'].transform('mean')

    test['Mean_IPAnnualReimbursementAmtPerBeneID']=test.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')


    #4Creating 15 new columns 'Mean_IPAnnualDeductibleAmt_PerColumn' for physicians, beneficiary, claim diagnosis and procedure codes.

    test['Mean_IPAnnualDeductibleAmt_PerAttendingPhysician']=test.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerOperatingPhysician']=test.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerOtherPhysician']=test.groupby('OtherPhysician')['IPAnnualDeductibleAmt'].transform('mean')

    test['Mean_IPAnnualDeductibleAmt_PerClmAdmitDiagnosisCode']=test.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerDiagnosisGroupCode']=test.groupby('DiagnosisGroupCode')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmDiagnosisCode_1']=test.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmDiagnosisCode_2']=test.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmDiagnosisCode_3']=test.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmDiagnosisCode_4']=test.groupby('ClmDiagnosisCode_4')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmDiagnosisCode_5']=test.groupby('ClmDiagnosisCode_5')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmDiagnosisCode_6']=test.groupby('ClmDiagnosisCode_6')['IPAnnualDeductibleAmt'].transform('mean')

    test['Mean_IPAnnualDeductibleAmt_PerClmProcedureCode_1']=test.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmProcedureCode_2']=test.groupby('ClmProcedureCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    test['Mean_IPAnnualDeductibleAmt_PerClmProcedureCode_3']=test.groupby('ClmProcedureCode_3')['IPAnnualDeductibleAmt'].transform('mean')

    test['Mean_IPAnnualDeductibleAmtPerBeneID']=test.groupby('BeneID')['IPAnnualDeductibleAmt'].transform('mean')


    #5Creating 15 new columns 'Mean_OPAnnualReimbursementAmt_PerColumn' for physicians, beneficiary, claim diagnosis and procedure codes.

    test['Mean_OPAnnualReimbursementAmt_PerAttendingPhysician']=test.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerOperatingPhysician']=test.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerOtherPhysician']=test.groupby('OtherPhysician')['OPAnnualReimbursementAmt'].transform('mean')

    test['Mean_OPAnnualReimbursementAmt_PerClmAdmitDiagnosisCode']=test.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerDiagnosisGroupCode']=test.groupby('DiagnosisGroupCode')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmDiagnosisCode_1']=test.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmDiagnosisCode_2']=test.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmDiagnosisCode_3']=test.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmDiagnosisCode_4']=test.groupby('ClmDiagnosisCode_4')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmDiagnosisCode_5']=test.groupby('ClmDiagnosisCode_5')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmDiagnosisCode_6']=test.groupby('ClmDiagnosisCode_6')['OPAnnualReimbursementAmt'].transform('mean')

    test['Mean_OPAnnualReimbursementAmt_PerClmProcedureCode_1']=test.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmProcedureCode_2']=test.groupby('ClmProcedureCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    test['Mean_OPAnnualReimbursementAmt_PerClmProcedureCode_3']=test.groupby('ClmProcedureCode_3')['OPAnnualReimbursementAmt'].transform('mean')

    test['Mean_OPAnnualReimbursementAmtPerBeneID']=test.groupby('BeneID')['OPAnnualReimbursementAmt'].transform('mean')


    #6)Creating 15 new columns 'Mean_OPAnnualDeductibleAmt_PerColumn' for physicians, beneficiary, claim diagnosis and procedure codes.

    test['Mean_OPAnnualDeductibleAmt_PerAttendingPhysician']=test.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerOperatingPhysician']=test.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerOtherPhysician']=test.groupby('OtherPhysician')['OPAnnualDeductibleAmt'].transform('mean')

    test['Mean_OPAnnualDeductibleAmt_PerClmAdmitDiagnosisCode']=test.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerDiagnosisGroupCode']=test.groupby('DiagnosisGroupCode')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmDiagnosisCode_1']=test.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmDiagnosisCode_2']=test.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmDiagnosisCode_3']=test.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmDiagnosisCode_4']=test.groupby('ClmDiagnosisCode_4')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmDiagnosisCode_5']=test.groupby('ClmDiagnosisCode_5')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmDiagnosisCode_6']=test.groupby('ClmDiagnosisCode_6')['OPAnnualDeductibleAmt'].transform('mean')

    test['Mean_OPAnnualDeductibleAmt_PerClmProcedureCode_1']=test.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmProcedureCode_2']=test.groupby('ClmProcedureCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    test['Mean_OPAnnualDeductibleAmt_PerClmProcedureCode_3']=test.groupby('ClmProcedureCode_3')['OPAnnualDeductibleAmt'].transform('mean')

    test['Mean_OPAnnualDeductibleAmtPerBeneID']=test.groupby('BeneID')['OPAnnualDeductibleAmt'].transform('mean')


    #doing one-hot encoding for Race and Gender.
    test=pd.get_dummies(test,columns=['Race','Gender'])

    #dropping unecessary columns
    test.drop(['Provider','BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt','AttendingPhysician', 'OperatingPhysician',
           'OtherPhysician', 'AdmissionDt', 'ClmAdmitDiagnosisCode', 'DischargeDt', 'DiagnosisGroupCode',
           'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
           'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
           'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
           'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
           'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
           'ClmProcedureCode_6', 'DOB', 'DOD', 'RenalDiseaseIndicator',
           'State', 'County','ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
           'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
           'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
           'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
           'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
           'ChronicCond_stroke','ClaimMinusAdmitDays'],axis=1,inplace=True)
    
    test['PotentialFraud']=0
    return test

#######################################################################################################################
def predictions(beneficiary,inpatient,outpatient,provider):
    patients=pd.concat([inpatient,outpatient])        #concatenates inpatient and outpatient dataframe on all the columns
    patientDetails=patients.merge(beneficiary,on='BeneID',how='inner')      #merges patient dataframe with beneficiary on Beneficiary ID.
    test=patientDetails.merge(provider,on='Provider',how='inner')    #merges previous dataframe with provider lables.
    test1=preprocessing(test)
    test2=feature_engineering(test1)
    #st.write(test2)
    exp=setup(data=test2,target='PotentialFraud', session_id=100,silent=True,html=False)
    _CWD = os.getcwd() 
    #filepath=os.path.join(_CWD,'model','lightgbm.pkl')
    #st.write(filepath,type(filepath))
    if not os.path.isfile(_CWD):
        url = r'https://github.com/prafulbatra/Fraudulent-Provider-Prediction/raw/main/Data/lightgbm.pkl'
        response = requests.get(url)													
        with open(os.path.join(_CWD,'lightgbm.pkl'), 'wb') as fopen:
            fopen.write(response.content)
    #with open(filepath, 'rb') as file:
    lightgbm=load_model('lightgbm')
    predictions=predict_model(lightgbm,data=test2)
    st.write(predictions)
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







