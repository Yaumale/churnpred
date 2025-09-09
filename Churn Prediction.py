import pandas as pd
import numpy as np

# from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier
import streamlit as st
import pickle

st.write('''
         Customer Churn Prediction
         ''')
st.sidebar.header("Please Input Customer Features")
def user_input():
    Tenure = st.sidebar.selectbox('Tenure',list(range(0,62)))
    Login_Device = st.sidebar.selectbox('Login Device',options=['Mobile Phone', 'Phone', 'Computer'])
    CityTier = st.sidebar.selectbox('City Tier',options=[1,2,3])
    WarehouseToHome = st.sidebar.number_input('Dist. From House To Warehouse',min_value=5,max_value=127,value=5)
    PreferredPaymentMode = st.sidebar.selectbox('Payment Mode',options=['Debit Card', 'UPI', 'Credit Card', 'COD', 'E wallet'])
    Gender = st.sidebar.selectbox('Gender',options=['Male','Female'])
    HourSpendOnApp = st.sidebar.number_input('Hours Spend On App',min_value=0,max_value=5,value=0)
    NumberOfDeviceRegistered = st.sidebar.number_input('Number Of Device Registered',min_value=1,max_value=6,value=1)
    PreferedOrderCat = st.sidebar.selectbox('Prefered Order Category',options=['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others','Fashion', 'Grocery'])
    SatisfactionScore = st.sidebar.number_input('Satisfaction Score',min_value=1,max_value=5,value=1)
    MaritalStatus = st.sidebar.selectbox('Marital Status',options=['Single', 'Divorced', 'Married'])
    NumberOfAddress = st.sidebar.number_input('Number Of Address',min_value=1,max_value=22,value=1)
    complain = st.sidebar.selectbox('Complain(Yes=1,No=0)',options=[0,1])
    OrderAmountHikeFromlastYear = st.sidebar.number_input('Order Amount Hike From Last Year',min_value=11,max_value=26,value=11)
    CouponUsed = st.sidebar.number_input('Coupon Used',min_value=0,max_value=16,value=0)
    OrderCount = st.sidebar.number_input('Order Count',min_value=1,max_value=16,value=1)
    DaySinceLastOrder = st.sidebar.number_input('Day Since Last Order',min_value=0,max_value=46,value=0)
    CashbackAmount = st.sidebar.number_input('Cashback Amount',min_value=0,max_value=325,value=0)

    data = ({
    'Tenure': Tenure,
    'PreferredLoginDevice': Login_Device,
    'CityTier': CityTier,
    'WarehouseToHome': WarehouseToHome,
    'PreferredPaymentMode': PreferredPaymentMode,
    'Gender': Gender,
    'HourSpendOnApp': HourSpendOnApp,
    'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
    'PreferedOrderCat': PreferedOrderCat,
    'SatisfactionScore': SatisfactionScore,
    'MaritalStatus': MaritalStatus,
    'NumberOfAddress': NumberOfAddress,
    'Complain': complain,
    'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
    'CouponUsed': CouponUsed,
    'OrderCount': OrderCount,
    'DaySinceLastOrder': DaySinceLastOrder,
    'CashbackAmount': CashbackAmount
})

    return pd.DataFrame([data])

df_predict = user_input()

st.subheader("Fitur yang Dimasukkan:")
st.write(df_predict)

model_loaded = pickle.load(open("Churn_Model_XGB.sav", "rb"))

prediksi = model_loaded.predict(df_predict)[0]

def hasil_predict():
    if prediksi == 0:
        return 'Not Churn'
    else:
        return 'Churn'
st.subheader("Churn Prediction:")
st.success(f"Churn prediction: {hasil_predict()}")