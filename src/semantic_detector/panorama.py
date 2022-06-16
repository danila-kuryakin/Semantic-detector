import time
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
  start_time = time.time()
  top, bot, left, right = 100, 100, 0, 500
  img1 = cv2.imread('../resources/dataset/BirdView/011---taiyuan/west_1.jpg')
  img2 = cv2.imread('../resources/dataset/BirdView/011---taiyuan/east_1.jpg')

  # img1 = imutils.rotate(img1, angle=-90)
  #img2 = imutils.rotate(img2, angle=180)


  srcImg = cv2.copyMakeBorder(img1, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  testImg = cv2.copyMakeBorder(img2, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

  img1 = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create()
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)
  # FLANN parameters
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)

  # Need to draw only good matches, so create a mask
  matchesMask = [[0, 0] for i in range(len(matches))]

  good = []
  pts1 = []
  pts2 = []
  # ratio test as per Lowe's paper
  for i, (m, n) in enumerate(matches):
    if m.distance < 0.6 * n.distance:
      good.append(m)
      pts2.append(kp2[m.trainIdx].pt)
      pts1.append(kp1[m.queryIdx].pt)
      matchesMask[i] = [1, 0]

  draw_params = dict(matchColor=(0, 255, 0),
                     singlePointColor=(255, 0, 0),
                     matchesMask=matchesMask,
                     flags=0)
  img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
  plt.imshow(img3, ), plt.show()
  # ДАЛЬШЕ-СЛОЖНО
  rows, cols = srcImg.shape[:2]
  MIN_MATCH_COUNT = 3
  if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    warpImg = cv2.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                  flags=cv2.WARP_INVERSE_MAP)

    for col in range(0, cols):
      if srcImg[:, col].any() and warpImg[:, col].any():
        left = col
        break
    for col in range(cols - 1, 0, -1):
      if srcImg[:, col].any() and warpImg[:, col].any():
        right = col
        break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
      for col in range(0, cols):
        if not srcImg[row, col].any():
          res[row, col] = warpImg[row, col]
        elif not warpImg[row, col].any():
          res[row, col] = srcImg[row, col]
        else:
          srcImgLen = float(abs(col - left))
          testImgLen = float(abs(col - right))
          alpha = srcImgLen / (srcImgLen + testImgLen)
          res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    # opencv is bgr, matplotlib is rgb
    cv2.imshow('flann_result', cv2.resize(res, (1200, 900)))
    cv2.imwrite('../../out/west+east.jpg', res)
    end_time = time.time()
    print(end_time - start_time)
    cv2.waitKey()
  else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
