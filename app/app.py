import streamlit as st
import numpy as np
from PIL import Image

import cv2
import numpy as np
import time
import tensorflow as tf

# st.set_page_config(page_title='NOE DEMO', layout="wide")
st.set_page_config(page_title='NOE DEMO')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css('./style.css')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.image("./logo-looping.png", use_column_width=True)

st.sidebar.image("./logo-looping.png", use_column_width=True)
st.sidebar.image("./umons.png", use_column_width=True)
st.sidebar.image("./click.png", use_column_width=True)

models = {'ResNet50': 180,
          'DenseNet121': 224,
          'MobileNet': 224,
          'MobileNet_V2': 224,
          'MobileNetV3Small': 224
        }
        
model_name = 'MobileNetV3Small'

classes = ['Bouchon de liege', 'Lunettes', 'Paquet de chips', 'Pile', 'Pot de yaourt']

if 'model' not in st.session_state:
    model = tf.keras.models.load_model('./MobileNetV3Small.h5')

    st.session_state.model = model

model = st.session_state.model

# if 'imsize' not in st.session_state:
#     if 'model' in st.session_state:
#         st.session_state

# model = tf.keras.models.load_model('./MobileNetV3Small.h5')

st.header('NOE - Shazam du Recyclage')
st.subheader('Upload Images')
uploaded_file = st.file_uploader('', type=['jpg','jpeg','png', 'JPG', 'JPEG'], accept_multiple_files=True)

if len(uploaded_file) > 0:
    for e, v in enumerate(uploaded_file):
        image = Image.open(v)
        cvimage = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cvimage = cv2.resize(cvimage, (models[model_name], models[model_name]))
        cvimage = np.expand_dims(cvimage, axis=0)
        t0 = round(time.time() * 1000)
        pred = model.predict(cvimage)[0]
        t1 = round(time.time() * 1000)
        # top_values_index = sorted(range(len(pred)), key=lambda i: pred[i])[-3:]
        top_values_index = np.argsort(-pred)[:3]
        st.subheader('Image ' + str(e+1) + '/' + str(len(uploaded_file)))
        st.image(image)
        st.text('Inference time: ' + str(t1-t0) + 'ms')
        st.text('TOP 3 Results:')
        # for idx in top_values_index:
        #     txt = classes[idx] + ', ' + '{:.2f}'.format(pred[idx]*100)
        #     st.text(txt)
        txt = classes[top_values_index[0]] + ', ' + '{:.2f}'.format(pred[top_values_index[0]]*100) + '%'
        st.markdown("<div><span class='highlight2 green'>" + txt +"</span></div>", unsafe_allow_html=True)
        st.markdown("")
        txt = classes[top_values_index[1]] + ', ' + '{:.2f}'.format(pred[top_values_index[1]]*100)
        st.markdown("<div><span class='highlight2 orange'>" + txt +"</span></div>", unsafe_allow_html=True)
        st.markdown("")
        txt = classes[top_values_index[2]] + ', ' + '{:.2f}'.format(pred[top_values_index[2]]*100)
        st.markdown("<div><span class='highlight2 yellow'>" + txt +"</span></div>", unsafe_allow_html=True)
        st.markdown("")

st.subheader('OR Take a picutre')
picture = st.camera_input('')

if picture:
    image = Image.open(picture)
    cvimage = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cvimage = cv2.resize(cvimage, (models[model_name], models[model_name]))
    cvimage = np.expand_dims(cvimage, axis=0)
    t0 = round(time.time() * 1000)
    pred = model.predict(cvimage)[0]
    t1 = round(time.time() * 1000)
    # top_values_index = sorted(range(len(pred)), key=lambda i: pred[i])[-3:]
    top_values_index = np.argsort(-pred)[:3]
    st.text('Inference time: ' + str(t1-t0) + 'ms')
    st.text('TOP 3 Results:')
    # for idx in top_values_index:
    #     txt = classes[idx] + ', ' + '{:.2f}'.format(pred[idx]*100)
    #     st.text(txt)
    txt = classes[top_values_index[0]] + ', ' + '{:.2f}'.format(pred[top_values_index[0]]*100) + '%'
    st.markdown("<div><span class='highlight2 green'>" + txt +"</span></div>", unsafe_allow_html=True)
    st.markdown("")
    txt = classes[top_values_index[1]] + ', ' + '{:.2f}'.format(pred[top_values_index[1]]*100)
    st.markdown("<div><span class='highlight2 orange'>" + txt +"</span></div>", unsafe_allow_html=True)
    st.markdown("")
    txt = classes[top_values_index[2]] + ', ' + '{:.2f}'.format(pred[top_values_index[2]]*100)
    st.markdown("<div><span class='highlight2 yellow'>" + txt +"</span></div>", unsafe_allow_html=True)
    st.markdown("")

