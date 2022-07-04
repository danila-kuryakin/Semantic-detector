import json

import cv2
import torch
import numpy as np

MODEL_PATH = '../resources/model/china_best.pt'


# load custom model
def get_yolov5():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)  # local repo
    model.conf = 0.3
    return model


def detect(img, view):
    model = get_yolov5()

    results = model(img)

    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    if view:
        marker_img = rectangle_selection(img, detect_res)
        cv2.imshow("marker image", cv2.resize(marker_img, (720, 480)))
    return detect_res

def rectangle_selection(img, segments):
    for segment in segments:
        if segment['class'] == 24:
            color = (0, 255, 255)
        elif segment['class'] == 11:
            color = (255, 255, 0)
        elif segment['class'] == 17:
            color = (255, 0, 255)
        elif segment['class'] == 22:
            color = (255, 127, 255)
        else:
            color = (0, 0, 255)
        point1 = (int(segment['xmin']), int(segment['ymin']))
        point2 = (int(segment['xmax']), int(segment['ymax']))

        cv2.rectangle(img, point1, point2, color, 2)
        label = '%s %.2f' % (segment['name'], segment['confidence'])

        cv2.putText(img, label, point1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return img

def polygon_selection(img, segments):
    for segment in segments:
        if segment['class'] == 24:
            color = (0, 255, 255)
        elif segment['class'] == 11:
            color = (255, 255, 0)
        elif segment['class'] == 17:
            color = (255, 0, 255)
        elif segment['class'] == 22:
            color = (255, 127, 255)
        else:
            color = (0, 0, 255)
            # print(element)
        point1 = segment['p1']
        point2 = segment['p2']
        point3 = segment['p3']
        point4 = segment['p4']
        pts = np.array([point1, point2, point3, point4], np.int32)

        cv2.polylines(img, [pts], True, color, 2)
        label = '%s %.2f' % (segment['name'], segment['confidence'])

        cv2.putText(img, label, point1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return img

