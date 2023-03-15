import numpy as np
from PIL import Image
import cv2
import streamlit as st
import base64
from itertools import chain

# import handle code
from src.yolov8.yolov8_predict import predict_yolov8
from src.craft.craft_predict import predict_craft
from src.craft.load_model import load_model_craft
from src.yolov8.load_model import load_model_yolov8
from src.utils.download_file import download_link
from src.utils.convert_format_bbox import convert_craft_to_rectangle, convert_xyxy2xywh, convert_xywh2xyxy


# ==========
trained_model = './weights/detect/craft_mlt_25k.pth'
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4 
cuda = False
canvas_size = 1280
mag_ratio = 1.5
poly = False
show_time = False
test_folder = 'upload_image.jpg'
refine  = False
refiner_model = 'weights/craft_refiner_CTW1500.pth'
color = {
    'Red': (255, 0, 0), 
    'Back': (0, 0, 0), 
    'Green': (0, 255, 0), 
    'Yellow': (255, 255, 0), 
    'Blue': (0, 0, 255)
}



# ======================  Load model  ================
with st.spinner("Loading Model....(50%)"):
    net, refine_net = load_model_craft()

with st.spinner("Loading Model....(100%)"):
    model_yolov8 = load_model_yolov8()



bg_image = st.sidebar.file_uploader("Upload image:", type=["png", "jpg", "jpeg"])
model_detect_name = st.sidebar.selectbox('Choose model for detection?', ['Craft', 'Yolov8', 'Best Model'])
threshold = st.sidebar.slider('Select a threshold:', 0.0, 1.0, 0.5)
bounding_box_color = st.sidebar.selectbox('Choose color to draw bounding box?', ['Red', 'Back', 'Green', 'Yellow', 'Blue'])



if bg_image: 
    # save image
    with open('upload_image.jpg', 'wb') as f: 
        f.write(bg_image.getbuffer())

    # show image
    show_image = Image.open(bg_image).resize((700, 500))
    st.image(show_image, caption='Input image')

    # detect phase
    main_image = cv2.cvtColor(cv2.imread('upload_image.jpg'), cv2.COLOR_BGR2RGB)
    if model_detect_name == 'Craft':
        boxes = predict_craft(net, refine_net, image_path='upload_image.jpg', text_threshold=threshold, cuda_state=False)
    elif model_detect_name == 'Yolov8':
        boxes = predict_yolov8(model_yolov8, image_path='upload_image.jpg', text_threshold=threshold)
    elif model_detect_name == "Best Model":
        # detect bang craft + yolo
        boxes_craft = predict_craft(net, refine_net, image_path='upload_image.jpg', text_threshold=threshold, cuda_state=False)
        boxes_craft = [convert_craft_to_rectangle(box) for box in boxes_craft]
        boxes_yolov8 = predict_yolov8(model_yolov8, image_path='upload_image.jpg', text_threshold=threshold)
        boxes = boxes_craft + boxes_yolov8

        # ap dung mns
        boxes = np.array([convert_xyxy2xywh(i) for i in boxes])
        score = [0.8] * len(boxes_craft) + [0.6] * len(boxes_yolov8)
        idx = cv2.dnn.NMSBoxes(boxes, score, score_threshold=0.4, nms_threshold=0.2)

        boxes = [convert_xywh2xyxy(i) for i in boxes[idx]]



    # rec phase
    txt_content = ''
    for count, box in enumerate(boxes): 
        # cat anh ra va convert to PIL
        if model_detect_name == 'Craft':
            # convert to int value and replace negative value to zero
            box = [list(map(int, i)) for i in box]
            box = [list(map(lambda x: max(x, 0) ,i)) for i in box]
            txt_content += f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]},{box[2][0]},{box[2][1]},{box[3][0]},{box[3][1]}\n"
            box = np.array(box).reshape((-1, 1, 2))

            main_image = cv2.polylines(main_image, [box], isClosed=True, color=color[bounding_box_color], thickness=int(0.01 * main_image.shape[0]))

        elif model_detect_name == 'Yolov8':
            txt_content += f"{box[0]},{box[1]},{box[2]},{box[3]}\n"
            main_image = cv2.rectangle(main_image, box[:2], box[-2:], color=color[bounding_box_color], thickness=int(0.01 * main_image.shape[0]))
        elif model_detect_name == 'Best Model':
            txt_content += f"{box[0]},{box[1]},{box[2]},{box[3]}\n"
            main_image = cv2.rectangle(main_image, box[:2], box[-2:], color=color[bounding_box_color], thickness=int(0.01 * main_image.shape[0]))



    # show image to screen
    main_image = cv2.resize(main_image, (700, 500), interpolation = cv2.INTER_AREA)
    st.image(main_image, caption='Output image')


    # button to download result
    download_button = st.sidebar.download_button(
        label="Download Result",
        data=txt_content,
        file_name="result.txt",
        mime="text/plain"
    )
    if download_button:
        st.markdown(download_link(txt_content, "result.txt", "text/plain"), unsafe_allow_html=True)


        
        
            
            
