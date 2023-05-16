from types import GeneratorType
import streamlit as st
import cv2
import numpy as np
from PIL import Image


filename = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(filename)

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    new_img = np.array(frame.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Face detection
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    #Draw rectangle
    for (x, y, width, height) in faces:
        cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 2)

    FRAME_WINDOW.image(img)