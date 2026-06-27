import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in the Titanic Journey")
pclass=st.slider("Enter the Passenger Class for the user",1,3)
sex=st.selectbox("Enter the Passenger Sex for the user",["male","female"])
sibsp=st.slider("Enter the number of Sibling/Spouse aboard",1,8)
parch=st.slider("Enter the number of Parents/Children aboard",0,6)
fare=st.number_input("Enter the Fare paid by the passenger")
embarked=st.selectbox("Enter the port of Embarkation for the user",["Southampton","Cherbourg","Queenstown"])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])
if st.button('Data'):
    st.write(data)

model=load_model("model.h5")
with open("label_encoder.pkl","rb") as f:
    label=pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler=pickle.load(f)

with open("onehot_encoder.pkl","rb") as f:
    onehot=pickle.load(f)

data['Sex']=label.transform(data['Sex'])
data['Embarked']=data['Embarked'].replace('Cherbourg', 'Chebourg')
embarked=onehot.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)
y=y[0][0]

def Chance(y):
    if y>0.5:
        return 'The Passenger has a high chance of survival'
    else:
        return 'The Passenger has a low chance of survival'

if st.button('Predict Survival Chance'):
    st.write('Probability of Passenger Survival Chance',y)
    st.write(Chance(y))