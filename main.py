import joblib
import streamlit as st
import pandas as pd

model=joblib.load("flower.pkl")

st.title("IRIS CLASSIFICATION FOR FLOWERS ")

def user_report():
    sepal_length=st.number_input("Sepal Length ")
    sepal_width=st.number_input("Sepal width ")
    petal_length=st.number_input("Petal Length ")
    petal_width=st.number_input("Petal width ")
   


    user_data={
    'sepal_length' : sepal_length,
    'sepal_width': sepal_width,
    'petal_length':  petal_length,
    'petal_width': petal_width
    
    
    }


    data=pd.DataFrame(user_data, index=[0])
    return data

data=user_report()


st.write(data)

flower=model.predict(data)
if st.checkbox("Show Result"):
    st.subheader(flower)
    








