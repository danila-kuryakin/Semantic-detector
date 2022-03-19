import cv2
import numpy as np
img1_square_corners = np.float32([[710, 240], [963, 234], [1502, 1058], [59, 1070]])
img2_quad_corners = np.float32([[670, 35], [895, 0], [910, 1080], [670, 1080]])
h, mask = cv2.findHomography(img1_square_corners, img2_quad_corners)
print(' mask:\n',mask)
print(' h:\n',h)
print(' h0:\n',h[0])
print(' h00:\n',h[0,0])
print(' h1:\n',h[1])
im = cv2.imread('../resources/dataset/BirdView/001---changzhou/north_1.jpg')
width, height, _ = im.shape
# print('{} {}'.format(width, height))
out = cv2.warpPerspective(im, h, (height, width))

cv2.imwrite('../../out/out.jpg', out)

im_r = cv2.resize(im, (height//2, width//2))
out_r = cv2.resize(out, (height//2, width//2))

cv2.imshow('im', im_r)
cv2.imshow('out', out_r)

cv2.waitKey()