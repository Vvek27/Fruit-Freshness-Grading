import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.elements.image import image_to_url
import streamlit_lottie
from streamlit_lottie import st_lottie
import requests

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import os,random
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(initial_sidebar_state="collapsed")

html_temp = """
    <div style="background-color:skyblue;padding:10px">
    <h2 style="color:black;text-align:center;">Fruit Freshness Grading</h2>
    </div>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://th.bing.com/th/id/R.856f31d9f475501c7552c97dbe727319?rik=Eq9oehb4QunXVw&riu=http%3a%2f%2fwww.baltana.com%2ffiles%2fwallpapers-5%2fWhite-Background-High-Definition-Wallpaper-16573.jpg&ehk=I38kgsJb2jc3ycTK304df0ig%2flhB3PaaXRrqcPVwDgA%3d&risl=&pid=ImgRaw&r=0")
 }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # sidebar
    st.set_option('deprecation.showPyplotGlobalUse', False)
    activities=["Select activity","Images","Filters","Prediction","About"]
    choice=st.sidebar.radio("",activities)
    if choice=="Select activity":
        activity()
    if choice=="Images":
        Images()
    if choice=="Filters":
        Filters()
    if choice=="Prediction":
        Prediction()
    if choice=="About":
        About()


def activity():
    def lottie_file(url:str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_hello=lottie_file("https://assets3.lottiefiles.com/packages/lf20_E3exCx.json")
    st_lottie( lottie_hello, speed=1, reverse=False,loop=True,quality="low",
    renderer="svg")
    
def Images():
    if st.checkbox("High Grade Images"):
        folder_n="app/fruit-freshness-grading/Train/A"
        a=random.choice(os.listdir(folder_n))
        b=random.choice(os.listdir(folder_n))
        c=random.choice(os.listdir(folder_n))
        random_n=[a,b,c]
        for img in random_n:
            img=folder_n + "/" + img
            img=image.load_img(img)
            st.image(img,width=300)
    if st.checkbox("Low Grade Images"):
        folder_n=r"app/fruit-freshness-grading/Train/L"
        a=random.choice(os.listdir(folder_n))
        b=random.choice(os.listdir(folder_n))
        c=random.choice(os.listdir(folder_n))
        random_n=[a,b,c]
        for i in range(len(random_n)):
            img=folder_n + "/" + random_n[i]
            img=image.load_img(img)
            st.image(img,width=300)
    
def Prediction():
    def classify(image,model):
        #load model
        my_model=load_model(model)
        prediction=my_model.predict(img)
        return prediction
    uploaded_file = st.file_uploader("Choose an banana image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        if st.checkbox("File information"):
            d={"File name":[uploaded_file.name],"File size":[uploaded_file.size],
            "File type":[uploaded_file.type]}
            df=pd.DataFrame(d)
            st.write(df)
        # save the particular file
        with open (uploaded_file.name,"wb") as f:
            f.write(uploaded_file.getbuffer())
        image_name=uploaded_file.name
        img_path=r"app/fruit-freshness-grading" + "/" + image_name
        img=image.load_img(img_path,target_size=(224,224))
        if st.checkbox("Show image"):
            st.image(img,width=400)
        img=image.img_to_array(img)/255
        img=np.array([img])
        result=classify(img,"vgg19model.h5")
        result=np.argmax(result)
        result=11-result
        if st.button("Classify"):
            if result>6:
                st.success("its healthy you can eat")
                st.write(result)
            else:
                st.warning("its not healthy dont eat")
                st.write(result)
def About():
    st.text("Made by vivek patel")
    st.text("2nd year M.Tech student at IIT kharagpur")
    st.text("Follow me on")
       
if __name__=="__main__":
    main()
