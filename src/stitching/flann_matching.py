import cv2
import imutils as imutils
import numpy as np

KNN = 2
LOWE = 0.7
TREES = 5
CHECKS = 50


def flann_matching(img1, img2):
  flann_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  flann_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create()
  flann_matcher = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': TREES}, {'checks': CHECKS})

  np.concatenate((flann_image, flann_image2), axis=0)

  keypoints_1, descriptiors1 = sift.detectAndCompute(flann_image, None)
  keypoints_2, descriptiors2 = sift.detectAndCompute(flann_image2, None)

  flann_matcher.knnMatch(descriptiors1, descriptiors2, k=KNN)
  flann_matches = flann_matcher.knnMatch(descriptiors1, descriptiors2, k=KNN)


  positive_for_flann = []

  for left_match, right_match in flann_matches:
    if left_match.distance < LOWE * right_match.distance:
      positive_for_flann.append(left_match)

  positive_for_flann = sorted(positive_for_flann, key=lambda x: x.distance)
  flann_result = cv2.drawMatches(flann_image, keypoints_1, flann_image2, keypoints_2, positive_for_flann[:], None,
                                 flags=2)
  cv2.imshow('flann_result', cv2.resize(flann_result, (1300, 700)))


def bf_matching(img1, img2):
  bf_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  bf_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create()
  cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

  np.concatenate((bf_image, bf_image2), axis=0)

  keypoints_1, descriptiors1 = sift.detectAndCompute(bf_image, None)
  keypoints_2, descriptiors2 = sift.detectAndCompute(bf_image2, None)

  bf_matcher = cv2.BFMatcher()
  bf_matches = bf_matcher.knnMatch(descriptiors1, descriptiors2, k=KNN)

  positive_for_bf = []
  for left_match, right_match in bf_matches:
    if left_match.distance < LOWE * right_match.distance:
      positive_for_bf.append(left_match)
  positive_for_bf = sorted(positive_for_bf, key=lambda x: x.distance)

  bf_result = cv2.drawMatches(bf_image, keypoints_1, bf_image2, keypoints_2, positive_for_bf[:], None, flags=2)
  cv2.imshow('bf_result', cv2.resize(bf_result, (1500, 700)))


if __name__ == '__main__':
  # FLANN MATCH FASTER THEN BRUTE FORCE, BUT WORSE. WE CAN PLAY WITH IT IN FUTURE.

  image = cv2.imread('../resources/dataset/BirdView/001---changzhou/south_1.jpg')
  image2 = cv2.imread('../resources/dataset/BirdView/001---changzhou/south_2.jpg')
  image = imutils.rotate(image, angle=180)
  flann_matching(image, image2)

  cv2.imshow('img1', cv2.resize(image, (400, 300)))
  cv2.imshow('img2', cv2.resize(image2, (400, 300)))
  # cv2.imshow('fun', cv2.resize(fun_combining_image, (1000, 400)))

  cv2.waitKey()
