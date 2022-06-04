import cv2
import numpy as np


image = cv2.imread('../resources/dataset/BirdView/005---lanzhou/west_1.jpg')
cv2.imshow("image", cv2.resize(image, (800, 450)))
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
# set lower and upper color limits
low_val = (0,100,6)
high_val = (170,255,255)
# Threshold the HSV image
mask = cv2.inRange(hsv, low_val,high_val)

# remove noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((8,8),dtype=np.uint8))

# close mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((20,20),dtype=np.uint8))
#cv2.imshow("Image0", cv2.resize(mask, (800, 450)))
# improve mask by drawing the convexhull
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(mask,[hull],0,(255), -1)
# erode mask a bit to migitate mask bleed of convexhull
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=np.ones((1,1),dtype=np.uint8))
cv2.imshow("Image", cv2.resize(mask, (800, 450)))
# remove this line, used to show intermediate result of masked road
road = cv2.bitwise_and(image, image,mask=mask)
cv2.imshow("road", cv2.resize(road, (800, 450)))
# apply mask to hsv image
road_hsv = cv2.bitwise_and(hsv, hsv,mask=mask)
# set lower and upper color limits
low_val = (0,141,6)
high_val = (170,255,255)
# Threshold the HSV image
mask2 = cv2.inRange(road_hsv, low_val,high_val)
# apply mask to original image
result = cv2.bitwise_and(image, image,mask=mask2)

#show image
# cv2.imshow("Result", cv2.resize(result, (1000, 563)))
# cv2.imshow("Road", cv2.resize(road, (800, 450)))
# cv2.imshow("Mask", cv2.resize(mask, (800, 450)))
# cv2.imshow("Image", cv2.resize(image, (800, 450)))


cv2.waitKey(0)
cv2.destroyAllWindows()