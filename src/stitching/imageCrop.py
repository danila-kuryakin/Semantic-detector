"""
Скрипт, который централизирует гомографию, обрезая пустоты снизу и приводит всё к FullHD разрешению
"""
import cv2
import imutils


def image_crop(img, path):
  full_path = path + img
  image = cv2.imread(full_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    image = image[y:y + h, x:x + w]
    image = imutils.resize(image, height=1080)
    new_width = (1920-image.shape[1])//2

    image = cv2.copyMakeBorder(image, 0, 0, new_width, new_width, cv2.BORDER_CONSTANT)
    break
  # cv2.imshow('end', image)

  cv2.imwrite(full_path, image)
  cv2.waitKey()


path = '../resources/dataset/HomographyView/013---yancheng/'

image_one = 'East_1080.jpg'
image_two = 'North_1080.jpg'
image_three = 'South_1080.jpg'
image_five = 'West_1080.jpg'

image_crop(image_one, path)
image_crop(image_two, path)
image_crop(image_three, path)
image_crop(image_five, path)