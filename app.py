import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

model = load_model('gen.keras')
def plot_images(num_images):
    if num_images == 0:
        st.warning('Please select the number of images to generate')
        return
    # if num_images == 1:
    #     fig, ax = plt.subplots()
    #     with st.spinner('Generating Images...'):
    #         img = model(tf.random.normal((1, 28)))
    #         ax.imshow(img[0])
    #         ax.axis('off')
    #         time.sleep(3)
    #     st.pyplot(fig)
    # else:
    with st.spinner('Generating Images...'): 
        imgs = model(tf.random.normal((num_images, 28)))

        fig, ax = plt.subplots(ncols=num_images, nrows=1)
        for i, image in enumerate(imgs):
            if num_images == 1:
                ax.imshow(image)
                ax.axis('off')
            else:
                ax[i].imshow(image)
                ax[i].axis('off')
        time.sleep(3)
    st.pyplot(fig)

st.title('Fashion Generative Adversarial Network')
# model = load_model('gen.keras')
# c1,c2 = st.columns([])
number  = 0
choice = None
with st.container(border=True):
    st.header('User Preferences')
    c1,c2 = st.columns([2,1])
    with c1:
        options = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        choice = st.selectbox('Select a Category', options)
    with c2:
        number = st.slider('Number of Images', 0, 10, 1)

with st.container(border=True):
    st.header('Generated Images')
    if choice is not None:
        plot_images(number)
    



