import cv2
import numpy as np

img=cv2.imread('test.jpg')

kernel1 = np.ones((3,5),np.uint8)
kernel2 = np.ones((9,9),np.uint8)

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('imgGray', imgGray)

imgBW=cv2.threshold(imgGray, 230, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow('imgBW', imgBW)

img1=cv2.erode(imgBW, kernel1, iterations=1)
cv2.imshow('img1', img1)

img2=cv2.dilate(img1, kernel2, iterations=3)
cv2.imshow('img2', img2)

img3 = cv2.bitwise_and(imgBW,img2)
cv2.imshow('img3', img3)

img3= cv2.bitwise_not(img3)

cv2.imshow('img33', img3)

img4 = cv2.bitwise_and(imgBW,imgBW,mask=img3)

cv2.imshow('img4', img4)

imgLines= cv2.HoughLinesP(img4,15,np.pi/180,10, minLineLength = 440, maxLineGap = 15)


for i in range(len(imgLines)):
    for x1,y1,x2,y2 in imgLines[i]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('Final Image with dotted Lines detected', img)
cv2.waitKey()