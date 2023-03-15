import pandas as pd
import numpy as np

from PIL import Image
import cv2


from src.utils.four_points_transform import four_points_transform

# for streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# for vietocr
from src.vietocr.vietocr_predict import predict_vietocr

# for parseq
from src.parseq.parseq_predict import predict_parseq


from src.parseq.load_model import load_model_parseq
from src.vietocr.load_model import load_model_vietocr

with st.spinner("Loading Model...."):
    detector=load_model_vietocr()

with st.spinner("Loading Model...."):
    parseq, img_transform=load_model_parseq(device='cpu')

drawing_mode = 'polygon'
bg_image = st.sidebar.file_uploader("Upload image:", type=["png", "jpg", "jpeg"])
model_name = st.sidebar.selectbox('Choose model?', ['Vietocr', 'Parseq'])
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

st.write('''Please only draw quadrilateral.
Left-click to add a point, right-click to close the polygon, double-click to remove the latest point.''')

if bg_image: 
    # save image
    with open('upload_image.jpg', 'wb') as f: 
        f.write(bg_image.getbuffer())

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=500,
    width=700,
    drawing_mode=drawing_mode,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True)
)

# Do something interesting with the image data and paths


if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    try:
        coor = objects['path'].values[-1]
        coor = [i[1:] for i in coor[:-1]]

        # crop image + convert to quadrilateral + convert to PIL format
        img = cv2.imread('upload_image.jpg')
        img = cv2.resize(img, (700, 500))
        box = np.array(coor, dtype='float32')    
        sub_img = four_points_transform(img, box)

        # convert to PIL format
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        sub_img = Image.fromarray(sub_img)

        # predict with vietocr
        if model_name == "Vietocr":
            (pred, prob) = predict_vietocr(detector, sub_img)
            st.write(f'Model Vietocr predict this text: "{pred}" with proba {round(prob * 100, 2)}%')

        # predict with parseq
        elif model_name == "Parseq":
            (pred, prob) = predict_parseq(parseq=parseq, img_transform=img_transform, image=sub_img, device='cpu')
            st.write(f'Model parseq predict this text: "{pred[0]}" with proba {round(prob * 100, 2)}%')

    except: 
        pass

    