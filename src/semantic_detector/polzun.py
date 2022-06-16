import cv2
import numpy as np


def nothing(x):
    pass


if __name__ == '__main__':

    # Load image
    image = cv2.imread('north_1.jpg')

    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('LMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('LMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 255)
    cv2.setTrackbarPos('LMax', 'image', 255)
    cv2.setTrackbarPos('SMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while 1:
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('LMin', 'image')
        vMin = cv2.getTrackbarPos('SMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('LMax', 'image')
        vMax = cv2.getTrackbarPos('SMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
            print("(hMin = %d , lMin = %d, sMin = %d), (hMax = %d , lMax = %d, sMax = %d)" % (
                hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', cv2.resize(result, (600, 300)))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey()
