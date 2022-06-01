import math

import cv2
import numpy
import numpy as np
import lineOperations

def realtime_semantic_detector_RGB(img, minLineLength, maxLineGap):
    threshold1 = 250
    threshold2 = 150

    line_rho = 1
    line_theta = 180
    line_threshold = 50

    deltaX = 1
    deltaY = 1

    dst = cv2.Canny(img, threshold1, threshold2, None, 3)
    # cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    # img_result = numpy.array(img)

    lines = cv2.HoughLinesP(dst, line_rho, np.pi / line_theta, line_threshold, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)

    # ones = np.ones(img.shape)
    zeros = np.zeros(img.shape)

    linesList = []
    linearEquation = []

    height, width, _ = img.shape
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 < height / 2 and y2 < height / 2:
                continue
            if abs(y1 - y2) < 10:
                if y1 < height // 9 and y2 < height // 9:
                    continue
                else:
                    # cv2.line(ones, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                    continue
            cv2.line(zeros, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

            if len(linesList) == 0:
                linesList.append(line)
            else:
                for l in linesList:
                    x1List, y1List, x2List, y2List = l[0]

                    dx11 = abs(x1 - x1List)
                    dx12 = abs(x1 - x2List)
                    dy11 = abs(y1 - y1List)
                    dy12 = abs(y1 - y2List)

                    dx21 = abs(x2 - x1List)
                    dx22 = abs(x2 - x2List)
                    dy21 = abs(y2 - y1List)
                    dy22 = abs(y2 - y2List)
                    if dx11 <= deltaX and dy11 <= deltaY or dx12 <= deltaX and dy12 <= deltaY:
                        cv2.circle(zeros, (x1, y1), 3, (0, 0, 255), -1)
                        print("V1: ", l, line, (dx11, dy11, dx12, dy12))
                    elif dx21 <= deltaX and dy21 <= deltaY or dx22 <= deltaX and dy22 <= deltaY:
                        cv2.circle(zeros, (x2, y2), 3, (255, 0, 0), -1)
                        print("V2: ", l, line, (dx21, dy21, dx22, dy22))
                linesList.append(line)


    return zeros
