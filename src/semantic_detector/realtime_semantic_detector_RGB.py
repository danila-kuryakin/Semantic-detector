import cv2
import numpy as np
from homography import lineOperations


class LinearEquation:

    def __init__(self):
        self.line = dict()

    def add_line(self, key, value):
        lines = []
        if list(filter(lambda l: key == l, self.line)) == []:
            self.line[key] = [value]
        else:
            coefficientB = self.line.get(key)
            coefficientB.append(value)
            self.line[key] = coefficientB

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

    # linesList = []
    linearEquation = LinearEquation()

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
            # linesList.append(line[0])
            equation = lineOperations.find_line_equation(line[0])
            # equation = linearEquation.add_line(lineOperations.find_line_equation(line[0]))
            linearEquation.add_line(equation['k'], equation['b'])

    ret_line = []
    for k in linearEquation.line.keys():
        if len(linearEquation.line[k]) < 10:
            continue
        sum_b = sum(linearEquation.line[k])/len(linearEquation.line[k])
        y1 = 0
        x1 = int((y1 - sum_b) / k)
        y2 = height
        x2 = int((y2 - sum_b) / k)
        # cv2.line(zeros, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        ret_line.append([x1, y1, x2, y2])

        # print(linearEquation.line[k])

    ret_line.append([0, int(height/2), width, int(height/2)])
    ret_line.append([0, int(height - (height/5)), width, int(height - (height/5))])

    for line in ret_line:
        x1, y1, x2, y2 = line
        print(x1, y1, x2, y2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("img", cv2.resize(img, (600, 500)))
    return zeros
