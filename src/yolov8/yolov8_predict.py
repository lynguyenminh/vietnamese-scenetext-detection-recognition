import numpy as np


def predict_yolov8(model, image_path, text_threshold=0.25) -> list:
    raw_information_predict = model.predict(source=image_path, conf=text_threshold)
    boxes = []
    for coordinate in zip(raw_information_predict[0].boxes.xyxy):
        boxes.append(coordinate[0].cpu().numpy().astype(int))
    
    return boxes
