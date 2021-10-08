from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

url = ("http://dl.dropboxusercontent.com/s/8enqf9v1s1fq9ty/mnist.pkl?raw=1")
filename = "export.pkl"
urlretrieve(url,filename)

uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)
    
    
    
else:
    image = Image.open(uploaded_file)
    img_array = PILImage.create((uploaded_file))
    self.learn_inference = load_learner(Path()/filename)
    pred, pred_idx, probs = self.learn_inference.predict(self.img)
    st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
    st.image(self.img.to_thumb(500,500), caption='Uploaded Image')
  

    
