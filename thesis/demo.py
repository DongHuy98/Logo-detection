
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from androguard.core.bytecodes.apk import APK

st.title("Logo detection with mmdetection + resnet_50")
st.header("mmdetection chỉ có thể sử dụng trên linux và macOS")
st.header("Logo detection Example")
st.text("Upload an Image for Logo detection")

class_list =[
   'Apple', 'BMW','Heineken','HP','Intel','Mini','Starbucks','Vodafone', 'Unknown', 'Ferrari'
]

resnet_model = tf.keras.models.load_model('D:\Ananconda\streamlit\model_resnet_dataset8.h5')
def detect_brand_probality(img):

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 32, 32, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (32, 32)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array
    classes = resnet_model.predict(data)

    # run the inference
    classes = list(classes[0])
    location = classes.index(max(classes))
    return class_list[location]

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose an image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded logo', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = detect_brand_probality(image)
    st.write(label)