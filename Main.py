import streamlit as st
import os
from PIL import Image
import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
import time

def prediction(image):
    classifier_model = "model.h5"
      
    model = load_model(r"C:\Users\Swift3\Documents\Mini_Project\HiRISE-Net-master\model.h5")
      
    test_image = image.resize((227,227))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 227.0
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.reshape(test_image, (-1, 227, 227, 1))
    class_names = {0: 'other',1: 'crater',2: 'dark_dune',3: 'streak',4: 'bright_dune',5: 'impact',6: 'edge'}

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

fig = plt.figure()
st.title('Mars Surface detection')
st.markdown("Prediction : Surface feature")


file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
class_btn = st.button("Classify")
if file_uploaded is not None:
    image = image.load_img(file_uploaded, target_size=(227,227))    
    # image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=227, )
    
if class_btn:
    if file_uploaded is None:
        st.write("Invalid command, please upload an image")
    else:
        with st.spinner('Model working....'):
            plt.imshow(image)
            plt.axis("off")
            predictions = prediction(image)
            time.sleep(1)
            st.success('Classified')
            st.write(predictions)
