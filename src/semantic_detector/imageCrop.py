"""
Скрипт, который централизирует гомографию, обрезая пустоты снизу
"""
import cv2

image = cv2.resize(cv2.imread('h_south_1.jpg.png'), (1920, 1080))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

for c in cnts:
  x, y, w, h = cv2.boundingRect(c)
  image = image[y:y + h, x:x + w]
  break

#cv2.imshow('end', image)
cv2.imwrite('south.jpg', image)
cv2.waitKey()
