import cv2

from realtime_semantic_detector_RGB import realtime_semantic_detector_RGB

if __name__ == '__main__':

    path = '../resources/dataset/BirdView/007---fengxiang'
    name = 'south_1.jpg'
    full_path = path + '/' + name

    img = cv2.imread(full_path)
    try:
        # bird_view = image_homography(img)
        # View image
        cv2.imshow('ld_' + name, cv2.resize(img, (400, 300)))
        cv2.imwrite('../../out/ld_' + name, img)

        # cv2.imshow('h_' + name, cv2.resize(bird_view, (400, 300)))
        # cv2.imwrite('../../out/h_' + name, bird_view)

        # Semantic detector
        th = realtime_semantic_detector_RGB(img)
    except Exception as e:
        print('Error: ', name)
        print(e)

    cv2.waitKey()