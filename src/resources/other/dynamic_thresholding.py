import cv2
import numpy
import numpy as np


def nothing(x):
    pass


def dynamic_thresholding(img):

    # settings
    line_rho = 1
    line_theta = 180
    line_threshold = 50

    minLineLength = 5
    maxLineGap = 10

    #gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian filtering
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #dynamic thresholding()
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find lines
    dst = cv2.Canny(th3, 0, 255, None, 3)
    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(dst, line_rho, np.pi / line_theta, line_threshold, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)

    #new canvas for drawing canvas
    ones = np.ones(img.shape)

    # sorting line
    height, width, _ = img.shape
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:
                if y1 < height // 9 and y2 < height // 9:
                    continue
                else:
                    cv2.line(ones, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                    continue
            cv2.line(ones, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Dynamic_thresholding', cv2.resize(th3, (400, 300)))
    cv2.imshow("Canny", cv2.resize(cdstP, (400, 300)))
    cv2.imshow("Lines", cv2.resize(ones, (600, 500)))


if __name__ == '__main__':
    path = '../dataset/BirdView/001---changzhou/south_1.jpg'
    # path = '../resources/dataset/crossroad3_with_sun/camera6_right.png'

    img = cv2.imread(path)
    dynamic_thresholding(img)
    cv2.waitKey()
