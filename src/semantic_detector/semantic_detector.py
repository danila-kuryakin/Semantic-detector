import cv2
import numpy as np

from realtime_semantic_detector_RGB import realtime_semantic_detector_RGB

if __name__ == '__main__':

    # path = '../resources/dataset/BirdView/009---qiaoxi'
    # name = 'south_1.jpg'
    # full_path = path + '/' + name

    full_path = '../resources/dataset/BirdView/009---qiaoxi/south_2.jpg'

    img = cv2.imread(full_path)
    try:
        # bird_view = image_homography(img)
        # View image
        # cv2.imshow('ld_' + name, cv2.resize(img, (400, 300)))
        # cv2.imwrite('../../out/ld_' + name, img)

        # cv2.imshow('h_' + name, cv2.resize(bird_view, (400, 300)))
        # cv2.imwrite('../../out/h_' + name, bird_view)

        zeros = np.zeros(img.shape)

        # Semantic detector
        lineImage = realtime_semantic_detector_RGB(img=img, minLineLength=10, maxLineGap=20)
        cv2.imshow("Lines", cv2.resize(lineImage, (600, 500)))

        # line_rho = 1
        # line_theta = 180
        # line_threshold = 50

        # lines = cv2.HoughLinesP(lineImage, line_rho, np.pi / line_theta, line_threshold, minLineLength=10,
        #                         maxLineGap=300)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(zeros, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

        # cv2.imshow("Lines2", cv2.resize(zeros, (600, 500)))
        cv2.waitKey()
    except Exception as e:
        # print('Error: ', name)
        print(e)