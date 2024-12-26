import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle


model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehotencoder_geo.pkl','rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',onehotencoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

geo_encoded = onehotencoder_geo.transform([input_data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehotencoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.drop('Geography',axis=1),geo_encoded_df],axis=1)
input_df_scaled = scaler.transform(input_data)

prediction = model.predict(input_df_scaled)
prediction_prob = prediction[0][0]

st.write(f"churn probabity :{prediction_prob}")

if prediction_prob > 0.5:
    st.write("customer will churn")
else:
    st.write("customer will not  churn")
