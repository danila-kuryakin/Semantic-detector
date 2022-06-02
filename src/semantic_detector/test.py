import cv2
import numpy as np

# Read image
img = cv2.imread('../../out/clear_merge2.jpg')
cv2.imshow('img', img)
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 30, 255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
cv2.imshow('result', img1_bg)

blank_image = np.zeros((600, 800, 3), np.uint8)
blank_image[:, 0:800] = (87, 88, 91)
blank_image = cv2.bitwise_and(blank_image, blank_image, mask=mask)

xarosh = cv2.add(img1_bg, blank_image)
cv2.imshow('final2', xarosh)


cv2.waitKey(0)

cv2.destroyAllWindows()
