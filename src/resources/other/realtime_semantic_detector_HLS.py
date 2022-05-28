import cv2
import numpy
import numpy as np


def nothing(x):
    pass


if __name__ == '__main__':
    # Load image

    path = '../dataset/BirdView/001---changzhou/'
    fileName = path + 'north_1.jpg'
    # fileName = '../resources/dataset/crossroad3/camera0_up.png'

    # path = '../../out/'
    # fileName = 'h_west_1.jpg'

    img = cv2.imread(fileName)

    # Create a window
    cv2.namedWindow('settings')
    # cv2.namedWindow('darkness')
    # cv2.namedWindow('gauss')
    # cv2.namedWindow('edges')
    # cv2.namedWindow('final')

    # cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv


    cv2.createTrackbar('alpha', 'settings', 0, 100, nothing)
    cv2.createTrackbar('beta', 'settings', 0, 100, nothing)
    cv2.createTrackbar('gamma', 'settings', 0, 100, nothing)
    cv2.setTrackbarPos('alpha', 'settings', 84)
    cv2.setTrackbarPos('beta', 'settings', 60)
    cv2.setTrackbarPos('gamma', 'settings', 0)

    cv2.createTrackbar('HMin', 'settings', 0, 255, nothing)
    cv2.createTrackbar('LMin', 'settings', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'settings', 0, 255, nothing)
    cv2.setTrackbarPos('HMin', 'settings', 0)
    cv2.setTrackbarPos('LMin', 'settings', 160)
    cv2.setTrackbarPos('SMin', 'settings', 0)

    cv2.createTrackbar('HMax', 'settings', 0, 255, nothing)
    cv2.createTrackbar('LMax', 'settings', 0, 255, nothing)
    cv2.createTrackbar('SMax', 'settings', 0, 255, nothing)
    cv2.setTrackbarPos('HMax', 'settings', 255)
    cv2.setTrackbarPos('LMax', 'settings', 255)
    cv2.setTrackbarPos('SMax', 'settings', 255)

    cv2.createTrackbar('threshold1', 'settings', 0, 500, nothing)
    cv2.createTrackbar('threshold2', 'settings', 0, 500, nothing)
    cv2.setTrackbarPos('threshold1', 'settings', 100)
    cv2.setTrackbarPos('threshold2', 'settings', 100)

    cv2.createTrackbar('line rho', 'settings', 1, 10, nothing)
    cv2.createTrackbar('line theta', 'settings', 1, 360, nothing)
    cv2.createTrackbar('line threshold', 'settings', 1, 500, nothing)
    cv2.createTrackbar('minLine', 'settings', 0, 200, nothing)
    cv2.createTrackbar('maxLine', 'settings', 0, 200, nothing)
    cv2.setTrackbarPos('line rho', 'settings', 1)
    cv2.setTrackbarPos('line theta', 'settings', 180)
    cv2.setTrackbarPos('line threshold', 'settings', 20)
    cv2.setTrackbarPos('minLine', 'settings', 5)
    cv2.setTrackbarPos('maxLine', 'settings', 10)

    # Initialize
    alpha = beta = gamma = 0
    pAlpha = pBeta = pGamma = 0
    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    threshold1 = threshold2 = 0
    pThreshold1 = pThreshold2 = 0

    line_rho = line_theta = line_threshold = minLineLength = maxLineGap = 0
    pLine_rho = pLine_theta = pLine_threshold = pMinLineLength = pMaxLineGap = 0

    cv2.imshow('settings', (800, 600))
    cv2.resizeWindow('settings', 400, 300)
    # cv2.imshow('settings')

    while 1:
        # Get current positions of all trackbars
        alpha = cv2.getTrackbarPos('alpha', 'settings')
        beta = cv2.getTrackbarPos('beta', 'settings')
        gamma = cv2.getTrackbarPos('gamma', 'settings')

        hMin = cv2.getTrackbarPos('HMin', 'settings')
        sMin = cv2.getTrackbarPos('LMin', 'settings')
        vMin = cv2.getTrackbarPos('SMin', 'settings')
        hMax = cv2.getTrackbarPos('HMax', 'settings')
        sMax = cv2.getTrackbarPos('LMax', 'settings')
        vMax = cv2.getTrackbarPos('SMax', 'settings')

        threshold1 = cv2.getTrackbarPos('threshold1', 'settings')
        threshold2 = cv2.getTrackbarPos('threshold2', 'settings')

        line_rho = cv2.getTrackbarPos('line rho', 'settings')
        line_theta = cv2.getTrackbarPos('line theta', 'settings')
        line_threshold = cv2.getTrackbarPos('line threshold', 'settings')
        minLineLength = cv2.getTrackbarPos('minLine', 'settings')
        maxLineGap = cv2.getTrackbarPos('maxLine', 'settings')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        darkGrey = cv2.addWeighted(gray, alpha*0.01, np.zeros(gray.shape, gray.dtype), beta*0.01, gamma*0.01)
        # darkGrey = cv2.addWeighted(gray, 0.4, np.zeros(gray.shape, gray.dtype), beta*0.01, 0)

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(darkGrey, darkGrey, mask=mask)

        gauss = cv2.GaussianBlur(result, (7, 7), cv2.BORDER_DEFAULT)
        edges = cv2.Canny(gauss, threshold1, threshold2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, line_rho, np.pi / line_theta, line_threshold, minLineLength=pMaxLineGap, maxLineGap=pMinLineLength)

        dst = cv2.Canny(img, threshold1, threshold2, None, 3)
        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        # Print if there is a change in HSV value
        if (pAlpha != alpha) | (pBeta != beta) | (pGamma != gamma) | (pThreshold1 != threshold1) | (pThreshold2 != threshold2) |\
                (pLine_rho != line_rho) | (pLine_theta != line_theta) | (pLine_threshold != line_threshold) | (pMaxLineGap != maxLineGap) | (pMinLineLength != minLineLength) | \
                (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
            print("(threshold1 = %d , threshold2 = %d, (alpha = %.2f , beta = %.2f, gamma = %.2f), (hMin = %d , lMin = %d, sMin = %d), (hMax = %d , lMax = %d, sMax = %d)" % (
                threshold1, threshold2, alpha*0.01, beta*0.01, gamma*0.01, hMin, sMin, vMin, hMax, sMax, vMax))
            pAlpha = alpha
            pBeta = beta
            pGamma = gamma

            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

            pThreshold1 = threshold1
            pThreshold2 = threshold2

            pLine_rho = line_rho
            pLine_theta = line_theta
            pLine_threshold = line_threshold
            pMinLineLength = minLineLength
            pMaxLineGap = maxLineGap

            ones = np.ones(img.shape)
            height, width, _ = img.shape
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 10:
                        if y1 < height // 9 and y2 < height // 9:
                            continue
                        else:
                            cv2.line(ones, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                            continue
                    cv2.line(ones, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('lines', cv2.resize(ones, (600, 500)))
            # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cv2.resize(cdstP, (600, 500)))

        # Display result image
        # cv2.imshow('darkness', cv2.resize(darkGrey, (400, 300)))
        # cv2.imshow('gauss', cv2.resize(gauss, (400, 300)))
        cv2.imshow('canny', cv2.resize(edges, (400, 300)))
        cv2.imshow('final', cv2.resize(result, (400, 300)))


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
    cv2.waitKey()