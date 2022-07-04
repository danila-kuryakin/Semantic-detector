import json

import cv2
from neural_network.segmentation import detect

if __name__ == '__main__':
    # img_path = '../resources/dataset/BirdView/004---jinchang/west_1.jpg'
    img_path = '../../out/h_west_1.png'
    img = cv2.imread(img_path)

    detect(img, True)


    cv2.waitKey(0)

