import numpy as np
from PIL import Image
import cv2
import streamlit as st

# import handle code
from src.utils.four_points_transform import four_points_transform
from src.utils.encode_base64 import encode_base64

from src.yolov8.yolov8_predict import predict_yolov8
from src.craft.craft_predict import predict_craft
from src.parseq.parseq_predict import predict_parseq
from src.vietocr.vietocr_predict import predict_vietocr

from src.craft.load_model import load_model_craft
from src.parseq.load_model import load_model_parseq
from src.vietocr.load_model import load_model_vietocr
from src.yolov8.load_model import load_model_yolov8
from src.utils.download_file import download_link
from src.utils.convert_format_bbox import convert_craft_to_rectangle, convert_xyxy2xywh, convert_xywh2xyxy
from src.utils.pre_process import histogram_equalzed_rgb



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



# ======================  Load model  ================
with st.spinner("Loading Model....(1/4)"):
    detector = load_model_vietocr()

with st.spinner("Loading Model....(2/4)"):
    parseq, img_transform = load_model_parseq(device='cpu')

with st.spinner("Loading Model....(3/4)"):
    net, refine_net = load_model_craft()

with st.spinner("Loading Model....(4/4)"):
    model_yolov8 = load_model_yolov8()



bg_image = st.sidebar.file_uploader("Upload image:", type=["png", "jpg", "jpeg"])
model_detect_name = st.sidebar.selectbox('Choose model for detection?', ['Craft', 'Yolov8', 'Best Model'])
model_rec_name = st.sidebar.selectbox('Choose model for recognization?', ['Vietocr', 'Parseq'])

if bg_image: 
    # save image
    with open('upload_image.jpg', 'wb') as f: 
        f.write(bg_image.getbuffer())

    # show image
    show_image = Image.open(bg_image).resize((700, 500))
    st.image(show_image, caption='Input image')


    # detect phase
    main_image = cv2.imread('upload_image.jpg')
    txt_content = ''

    if model_detect_name == 'Craft':
        boxes = predict_craft(net, refine_net, image_path='upload_image.jpg', text_threshold=0.65, cuda_state=False)
    elif model_detect_name == 'Yolov8':
        boxes = predict_yolov8(model_yolov8, image_path='upload_image.jpg', text_threshold=0.5)
    elif model_detect_name == "Best Model":
        # detect bang craft + yolo
        boxes_craft = predict_craft(net, refine_net, image_path='upload_image.jpg', text_threshold=0.65, cuda_state=False)
        boxes_craft = [convert_craft_to_rectangle(box) for box in boxes_craft]
        boxes_yolov8 = predict_yolov8(model_yolov8, image_path='upload_image.jpg', text_threshold=0.5)
        boxes = boxes_craft + boxes_yolov8

        # ap dung mns
        boxes = np.array([convert_xyxy2xywh(i) for i in boxes])
        score = [0.8] * len(boxes_craft) + [0.6] * len(boxes_yolov8)
        idx = cv2.dnn.NMSBoxes(boxes, score, score_threshold=0.4, nms_threshold=0.2)

        boxes = [convert_xywh2xyxy(i) for i in boxes[idx]]


    table = '''| Crop images  | Texts |\n| ------------- |:-------------:|\n'''

    # rec phase
    for count, box in enumerate(boxes): 
        # cat anh ra va convert to PIL
        if model_detect_name == 'Craft':
            sub_img = four_points_transform(main_image, np.array(box, dtype='float32'))
            # write content
            box = [list(map(int, i)) for i in box]
            box = [list(map(lambda x: max(x, 0) ,i)) for i in box]
            txt_content += f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]},{box[2][0]},{box[2][1]},{box[3][0]},{box[3][1]}\t"
            
        elif model_detect_name == 'Yolov8' or model_detect_name == 'Best Model':
            txt_content += f"{box[0]},{box[1]},{box[2]},{box[3]}\t"
            sub_img = main_image[box[1]:box[3], box[0]:box[2]]

        # ap dung can bang sang
        # sub_img = histogram_equalzed_rgb(sub_img)
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        sub_img = Image.fromarray(sub_img)

        # predict with vietocr
        if model_rec_name == "Vietocr":
            # predict
            (pred, prob) = predict_vietocr(detector, sub_img)
            # encode image to display
            encode = encode_base64(sub_img.resize((150, 60)))
            table += f'''| ![image{count}.png](data:image/png;base64,{encode})   | {pred}     |\n'''
            # write to txt
            txt_content += f"{pred}, {prob}\n"

        # predict with parseq
        elif model_rec_name == "Parseq":
            # predict
            (pred, prob) = predict_parseq(parseq=parseq, img_transform=img_transform, image=sub_img, device='cpu')
            # encode image to display
            encode = encode_base64(sub_img.resize((150, 60)))
            table += f'''| ![image{count}.png](data:image/png;base64,{encode})   | {pred[0]}     |\n'''
            # write to txt
            txt_content += f"{pred}, {prob}\n"
            
            

    st.markdown(table, unsafe_allow_html=True)

    # button to download result
    download_button = st.sidebar.download_button(
        label="Download Result",
        data=txt_content,
        file_name="result.txt",
        mime="text/plain"
    )
    if download_button:
        st.markdown(download_link(txt_content, "result.txt", "text/plain"), unsafe_allow_html=True)