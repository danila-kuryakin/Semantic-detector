import cv2
import numpy
import numpy as np


# path = '../resources/dataset/BirdView/001---changzhou/'
# file_name = 'south_1.jpg'

path = '../../out/'
file_name = 'h_west_1.jpg'

# path = '../resources/'
# file_name = 'test.png'
full_path = path + file_name

img = cv2.imread(full_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,
                           (15, 15), 2)

cv2.imshow("blurred", cv2.resize(blurred, (400, 300)))

ret, thresh = cv2.threshold(blurred,
                            130, 255,
                            cv2.THRESH_BINARY)

cv2.imshow("thresh", cv2.resize(thresh, (400, 300)))

dst = cv2.Canny(thresh, 50, 200, None, 3)
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

cv2.imshow("dst", cv2.resize(dst, (400, 300)))
cv2.imshow("cdstP", cv2.resize(cdstP, (400, 300)))

contours, hier = cv2.findContours(thresh.copy(),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
boxs = []
for c in contours:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 50 or cv2.contourArea(c) > 3000:
        continue

    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    boxs.append(box)

    cv2.circle(img, box[0], 3, (0, 0, 255), -1)
    cv2.circle(img, box[1], 3, (0, 255, 0), -1)
    cv2.circle(img, box[2], 3, (255, 0, 0), -1)
    cv2.circle(img, box[3], 3, (255, 0, 255), -1)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.imshow("contours", cv2.resize(img, (900, 700)))

while True:
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()