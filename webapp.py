from pathlib import WindowsPath
from numpy.lib import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.elements.image import image_to_url
from streamlit.util import _maybe_tuple_to_list
import streamlit_lottie
from streamlit_lottie import st_lottie
import requests
import cv2

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

def Filters():
    if st.checkbox("High Grade Image"):
        path_f=r"C:\Users\Hp\Desktop\Visual studio\Fruit Freshness Grading\Train\A"
        img_nm=random.choice(os.listdir(path_f))
        img_path=path_f + "/" + img_nm
        image=cv2.imread(img_path)
        width,height=1000,1000
        img_size=cv2.resize(image,(width,height))
        sobel_x=cv2.Sobel(img_size,-1,1,0)
        sobel_y=cv2.Sobel(img_size,-1,0,1)
        canny_img=cv2.Canny(img_size,80,150)
        st.header("Original Image")
        plt.matshow(img_size)
        st.pyplot()
        st.header("Gradient Sobel X Image")
        plt.matshow(sobel_x)
        st.pyplot()
        st.header("Gradient Sobel Y Image")
        plt.matshow(sobel_y)
        st.pyplot()
        st.header("Canny Image")
        plt.matshow(canny_img)
        st.pyplot()
    if st.checkbox("Low Grade Image"):
        path_file=r"C:\Users\Hp\Desktop\Visual studio\Fruit Freshness Grading\Train\L"
        img_name=random.choice(os.listdir(path_file))
        img_path=path_file + "/" + img_name
        image=cv2.imread(img_path)
        width,height=1000,1000
        img_size=cv2.resize(image,(width,height)) 
        sobel_x=cv2.Sobel(img_size,-1,1,0)
        sobel_y=cv2.Sobel(img_size,-1,0,1)
        canny_img=cv2.Canny(img_size,80,150)
        st.header("Original Image")
        plt.matshow(img_size)
        st.pyplot()
        st.header("Gradient Sobel X Image")
        plt.matshow(sobel_x)
        st.pyplot()
        st.header("Gradient Sobel Y Image")
        plt.matshow(sobel_y)
        st.pyplot()
        st.header("Canny Image")
        plt.matshow(canny_img)
        st.pyplot()

def Images():
    if st.checkbox("High Grade Images"):
        folder_n=r"C:\Users\Hp\Desktop\Visual studio\Fruit Freshness Grading\Train\A"
        a=random.choice(os.listdir(folder_n))
        b=random.choice(os.listdir(folder_n))
        c=random.choice(os.listdir(folder_n))
        random_n=[a,b,c]
        for img in random_n:
            img=folder_n + "/" + img
            img=image.load_img(img)
            st.image(img,width=300)
    if st.checkbox("Low Grade Images"):
        folder_n=r"C:\Users\Hp\Desktop\Visual studio\Fruit Freshness Grading\Train\L"
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
            d={"File name":[uploaded_file.name],"File size":[(uploaded_file.size)/10**6],
            "File type":[uploaded_file.type]}
            df=pd.DataFrame(d)
            st.write(df)
        # save the particular file
        with open (uploaded_file.name,"wb") as f:
            f.write(uploaded_file.getbuffer())
        image_name=uploaded_file.name
        img_path=r"C:\Users\Hp\Desktop\Visual studio" + "/" + image_name
        img=image.load_img(img_path,target_size=(224,224))
        if st.checkbox("Show image"):
            st.image(img,width=400)
        img=image.img_to_array(img)/255
        img=np.array([img])
        result1=11-np.argmax(classify(img,"vgg19model.h5"))
        result2=11-np.argmax(classify(img,"incepv3model.h5"))
        result3=11-np.argmax(classify(img,"xcepmodel.h5"))
        result4=(result1 + result2 + result3)/3
        model_list=["Select Model","VGG19 model","Inceptionv3 Model","Xception Model","Combined"]
        choice=st.selectbox("",model_list)
        if choice=="VGG19 model":
            if st.button("Classify"):
                if result1>6:
                    st.success("its healthy you can eat")
                    st.write("Freshness Level-->",result1)
                else:
                    st.warning("its not healthy don't eat")
                    st.write("Freshness Level-->",result1)
        elif choice=="Inceptionv3 Model":
            if st.button("Classify"):
                if result2>6:
                    st.success("its healthy you can eat")
                    st.write("Frehness Level-->",result2)
                else:
                    st.warning("its not healthy don't eat")
                    st.write("Freshness Level-->",result2)
        elif choice=="Xception Model":
            if st.button("Classify"):
                if result3>6:
                    st.success("its healthy you can eat")
                    st.write("Freshness Level-->",result3)
                else:
                    st.warning("its not healthy don't eat")
                    st.write("Freshness Level-->",result3)
        elif choice=="Combined":
            if st.button("Classify"):
                if result4>6:
                    st.success("its healthy you can eat")
                    st.write("Freshness Level-->",result4)
                else:
                    st.warning("its not healthy don't eat")
                    st.write("Freshness Level-->",result4)

def About():
    st.markdown('<h3 style="text-align:center;">Made By <span style="color:#4f9bce;font-weight:bolder;font-size:40px;">"vivek"</span></h3>',unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;text-decoration:none;font-weight:bolder;"><a style="text-decoration:none;color:rgb(90, 235, 133);" href="https://github.com/Vvek27">-> GitHub <-</a></h2>',unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;text-decoration:none;font-weight:bolder;"><a style="text-decoration:none;color:red;"href="https://mail.google.com/mail/u/0/#inbox?compose=DmwnWrRmTWjqWjGLDZjgrQbrlNkwCVzqjlCqlzBZkkLhnmzkHNVTRSMFNLNQWxplJGVnXJdVcNkL">-> Contact Me <-</a></h2>',unsafe_allow_html=True)

if __name__=="__main__":
    main()
