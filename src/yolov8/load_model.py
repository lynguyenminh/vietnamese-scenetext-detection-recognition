import streamlit as st
from ultralytics import YOLO


@st.cache(allow_output_mutation=True)
def load_model_yolov8():
    model = YOLO(f'weights/detect/model_yolov8.pt')
    return model

