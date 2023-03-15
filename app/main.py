#C:\\Users\karti\\PycharmProjects\\Deep Learning\\LipNet\\data\\s1\\bbaf2n.mpg
import streamlit as st
import os 
import imageio
import tensorflow as tf
from utils import load_data,num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')

#Setup the sidebar
# Setup the sidebar
with st.sidebar: 
    st.image("./ai1.gif")
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title("LipNet Full Stack App")
options = os.listdir(os.path.join('..','data','s1'))
#options = os.listdir('C:\\Users\\karti\\PycharmProjects\\Deep Learning\\LipNet\\data\\s1')
selected_video = st.selectbox('Choose video', options)
col1,col2 = st.columns(2)

if options:
    with col1:
        file_path = os.path.join('..','data','s1',selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video = open('test_video.mp4','rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2:
        st.info("This is animation video of input")
        video,annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif',video,fps=10)
        st.image('animation.gif',width=400)

        model = load_model()
        yhat = model.predict(tf.expand_dims(video,axis=0))
        st.info("Raw Output given by Model")
        st.text(tf.argmax(yhat,axis=1).numpy())
        st.info("After passing through CTC Decoder")
        decoder = tf.keras.backend.ctc_decode(yhat,[75],greedy=True)[0][0].numpy()
        st.text(decoder)
        st.info("Decode the raw tokens into words")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        

        


