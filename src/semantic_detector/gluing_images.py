import cv2
import imutils as imutils
import numpy as np
from PIL import Image

if __name__ == '__main__':
    image = cv2.imread('../resources/dataset/BirdView/013---yancheng/north.jpg')
    image2 = cv2.imread('../resources/dataset/BirdView/013---yancheng/south.jpg')

    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    image = imutils.rotate(image, angle=180)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    fun_combining_image = np.concatenate((image, image2), axis=0)
    keypoints_1, descriptiors1 = sift.detectAndCompute(image, None)
    keypoints_2, descriptiors2 = sift.detectAndCompute(image2, None)

    left_matches = sift.detectAndCompute(image, None)
    right_matches = sift.detectAndCompute(image2, None)

    matches = bf.match(descriptiors1, descriptiors2)
    matches = sorted(matches, key=lambda x: x.distance)

    resultImage = cv2.drawMatches(image, keypoints_1, image2, keypoints_2, matches[:], None, flags=2)

    cv2.imshow('img1', cv2.resize(image, (400, 300)))
    cv2.imshow('img2', cv2.resize(image2, (400, 300)))

    cv2.imshow('img3', cv2.resize(resultImage, (1000, 400)))
    cv2.imshow('test', cv2.resize(fun_combining_image, (1000, 400)))
    KNN = 2
    LOWE = 0.7
    TREES = 5
    CHECKS = 50

    matcherHabr = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': TREES}, {'checks': CHECKS})
    matchesHabr = matcherHabr.knnMatch(descriptiors1, descriptiors2, k=KNN)

    positive = []
    for left_match, right_match in matchesHabr:
        if left_match.distance < LOWE * right_match.distance:
            positive.append(left_match)

    positive = sorted(positive, key=lambda x: x.distance)

    imageHabr = cv2.drawMatches(image, keypoints_1, image2, keypoints_2, positive[:], image)

    cv2.imshow('imageHabr', cv2.resize(imageHabr, (1500, 800)))

    cv2.waitKey()
