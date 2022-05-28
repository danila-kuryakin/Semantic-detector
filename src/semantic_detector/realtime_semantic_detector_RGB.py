import math

import cv2
import numpy
import numpy as np


def nothing(x):
    pass

def realtime_semantic_detector_RGB(img):

    # Create a window
    cv2.namedWindow('settings')

    cv2.createTrackbar('threshold1', 'settings', 0, 400, nothing)
    cv2.createTrackbar('threshold2', 'settings', 0, 400, nothing)
    cv2.setTrackbarPos('threshold1', 'settings', 250)
    cv2.setTrackbarPos('threshold2', 'settings', 150)

    cv2.createTrackbar('line rho', 'settings', 1, 10, nothing)
    cv2.createTrackbar('line theta', 'settings', 1, 360, nothing)
    cv2.createTrackbar('line threshold', 'settings', 1, 500, nothing)
    cv2.createTrackbar('minLine', 'settings', 0, 200, nothing)
    cv2.createTrackbar('maxLine', 'settings', 0, 200, nothing)
    cv2.setTrackbarPos('line rho', 'settings', 1)
    cv2.setTrackbarPos('line theta', 'settings', 180)
    cv2.setTrackbarPos('line threshold', 'settings', 50)
    cv2.setTrackbarPos('minLine', 'settings', 5)
    cv2.setTrackbarPos('maxLine', 'settings', 10)

    # Initialize
    alpha = beta = gamma = 0
    pAlpha = pBeta = pGamma = 0

    threshold1 = threshold2 = 0
    pThreshold1 = pThreshold2 = 0

    line_rho = line_theta = line_threshold = minLineLength = maxLineGap = 0
    pLine_rho = pLine_theta = pLine_threshold = pMinLineLength = pMaxLineGap = 0

    cv2.imshow('settings', (800, 600))
    cv2.resizeWindow('settings', 400, 300)
    # cv2.imshow('settings')

    while 1:
        threshold1 = cv2.getTrackbarPos('threshold1', 'settings')
        threshold2 = cv2.getTrackbarPos('threshold2', 'settings')

        line_rho = cv2.getTrackbarPos('line rho', 'settings')
        line_theta = cv2.getTrackbarPos('line theta', 'settings')
        line_threshold = cv2.getTrackbarPos('line threshold', 'settings')
        minLineLength = cv2.getTrackbarPos('minLine', 'settings')
        maxLineGap = cv2.getTrackbarPos('maxLine', 'settings')

        dst = cv2.Canny(img, threshold1, threshold2, None, 3)
        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        # Print if there is a change in HSV value
        if (pAlpha != alpha) | (pBeta != beta) | (pGamma != gamma) | (pThreshold1 != threshold1) | \
            (pThreshold2 != threshold2) | (pLine_rho != line_rho) | (pLine_theta != line_theta) | \
            (pLine_threshold != line_threshold) | (pMaxLineGap != maxLineGap) | (pMinLineLength != minLineLength):

            print("(threshold1 = %d , threshold2 = %d, (alpha = %.2f , beta = %.2f, gamma = %.2f)" % (
                threshold1, threshold2, alpha*0.01, beta*0.01, gamma*0.01))
            pAlpha = alpha
            pBeta = beta
            pGamma = gamma

            pThreshold1 = threshold1
            pThreshold2 = threshold2

            pLine_rho = line_rho
            pLine_theta = line_theta
            pLine_threshold = line_threshold
            pMinLineLength = minLineLength
            pMaxLineGap = maxLineGap

            img_result = numpy.array(img)

            lines = cv2.HoughLinesP(dst, line_rho, np.pi / line_theta, line_threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

            ones = np.ones(img.shape)

            vertical_line = []
            height, width, _ = img.shape
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 10:
                        if y1 < height//9 and y2 < height//9:
                            continue
                        else:
                            cv2.line(ones, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                            continue
                    cv2.line(ones, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

            # cv2.imshow('image', cv2.resize(img_result, (600, 500)))
            cv2.imshow("Canny", cv2.resize(cdstP, (400, 300)))
            cv2.imshow("Lines", cv2.resize(ones, (600, 500)))

        # Display result image
        # cv2.imshow('darkness', cv2.resize(darkGrey, (400, 300)))
        # cv2.imshow('gauss', cv2.resize(gauss, (400, 300)))
        # cv2.imshow('edges', cv2.resize(edges, (400, 300)))
        # cv2.imshow('final', cv2.resize(result, (400, 300)))


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey()