import cv2
import numpy
import numpy as np


def region_of_interest(img, vertices):
  mask = np.zeros_like(img)
  channel_count = img.shape[2]
  match_mask_color = (255,) * channel_count
  cv2.fillPoly(mask, vertices, match_mask_color)
  masked_image = cv2.bitwise_and(img, mask)
  return masked_image


#full_path = 'h_west_1.jpg (2).png'
full_path = '../resources/dataset/BirdView/001---changzhou/east.jpg'
img = cv2.imread(full_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (15, 15), 2)

ret, thresh = cv2.threshold(gray, 150, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
cv2.imshow("thresh", cv2.resize(thresh, (800, 600)))

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
find_contours = contours, hierarchy
find_contours = find_contours[0] if len(find_contours) == 2 else find_contours[1]


image_with_contours = img.copy()
good_contours = []
for c in find_contours:
  area = cv2.contourArea(c)
  if area > 100:
    cv2.drawContours(image_with_contours, [c], -1, (0, 0, 255), 2)
    good_contours.append(c)


contours_combined = np.vstack(good_contours)

cv2.imshow("contours", cv2.resize(image_with_contours, (800, 600)))

result = img.copy()
hull = cv2.convexHull(contours_combined)
cv2.polylines(result, [hull], True, (0, 0, 255), 2)
cv2.imshow("result", cv2.resize(result, (800, 600)))

region_of_interest_vertices = []
for c in hull:
  region_of_interest_vertices.append((c[0][0], c[0][1]))

image = img.copy()
cropped_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))

cv2.imshow("cropped_image", cv2.resize(cropped_image, (800, 600)))

dst = cv2.Canny(thresh, 50, 150)
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

cv2.imshow("dst", cv2.resize(dst, (800, 600)))
cv2.imshow("cdstP", cv2.resize(cdstP, (800, 600)))

cv2.waitKey()
