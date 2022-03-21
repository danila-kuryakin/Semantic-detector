import cv2
import numpy as np
import matplotlib.pyplot as plt

# Searches for linear equation variables
# y = kx + b
from cv2 import CV_32F, flann


def find_line_equation(line):
    x1, y1, x2, y2 = line
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return {'k': k, 'b': b}


# Looking for the point where the lines intersect
def find_intersection_point(line_a, line_b):
    x = int((line_b['b'] - line_a['b'])/(line_a['k'] - line_b['k']))
    y = int(line_b['k'] * x + line_b['b'])
    return x, y

def apply_threshold(filtered):
    ret, thresh = cv2.threshold(filtered, 254, 255, cv2.THRESH_OTSU)
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    #plt.title('After applying OTSU threshold')
    #plt.show()
    return thresh

def line_detection(img):
    height, width, _ = img.shape

    #cv2.imshow('img', cv2.resize(img, (800, 600)))

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', cv2.resize(gray, (800, 600)))

    #adjust contrast
    a = 5
    b= -500
    darken = cv2.addWeighted(gray, a, np.zeros(gray.shape,gray.dtype), 0, b)
    #cv2.imshow('darken', cv2.resize(darken, (800, 600)))

    # bluring image
    Blurrr = cv2.GaussianBlur(darken,(5,5),cv2.BORDER_DEFAULT)
    #cv2.imshow('Blurrr', cv2.resize(Blurrr, (800, 600)))

    # little moooore contrast
    a = 2
    b = -100
    darken2 = cv2.addWeighted(Blurrr, a, np.zeros(gray.shape, gray.dtype), 0, b)
    #cv2.imshow('darken2', cv2.resize(darken2, (800, 600)))


    #################
    # Canny filter
    edges = cv2.Canny(darken2, 100, 300, apertureSize=3)
    #cv2.imshow('edges', cv2.resize(edges, (800, 600)))

    #trsh = apply_threshold(gray)####
   # cv2.imshow('trsh', cv2.resize(trsh, (800, 600)))

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 250, minLineLength=height-height//5, maxLineGap=height)

    left_line = [width, 0, width, height]
    right_line = [0, 0, 0, height]

    down_line = [0, height, width, height]

    for line in lines:

        x1, y1, x2, y2 = line[0]
        # Classification vertical lines
        if y2 > height-height//4 and y1 < height//4:
            # View lines
          #  cv2.line(img, (x1, y1), (x2, y2), (0, 0, 150), 1)

            if x2 > right_line[2]:
                right_line = [x1, y1, x2, y2]
            continue
        if y1 > height-height//4 and y2 < height//4:
            # View lines
        #    cv2.line(img, (x2, y2), (x1, y1), (0, 150, 150), 1)

            if x1 < left_line[2]:
                left_line = [x2, y2, x1, y1]
            continue

        # TODO: redo please
        # Classification horizontal lines
        if y1 > height//2 and y2 > height//2:
            # View lines
         #   cv2.line(img, (x1, y1), (x2, y2), ( 150, 0,0), 1)
            if y1 < down_line[1] and y2 < down_line[3]:
                down_line = [x1, y1, x2, y2]
            continue

    # TODO: bad idea
    # Find up line
    up_line = (down_line[0], down_line[1] - height // 2, down_line[2], down_line[3] - height // 2)# start_width start_heigh end_width end_heigh

    return left_line, right_line, up_line, down_line


def point_detection(left_line, right_line, up_line, down_line):
    # Calculation of the equation of a straight line
    right_line_equation = find_line_equation(right_line)
    left_line_equation = find_line_equation(left_line)
    down_line_equation = find_line_equation(down_line)
    up_line_equation = find_line_equation(up_line)

    # Calculation point
    point_rd = find_intersection_point(right_line_equation, down_line_equation)
    point_ru = find_intersection_point(right_line_equation, up_line_equation)
    point_ld = find_intersection_point(left_line_equation, down_line_equation)
    point_lu = find_intersection_point(left_line_equation, up_line_equation)

    return point_lu, point_ru, point_rd, point_ld


def image_homography(img):
    # TODO: remove magic
    magic_indent = 150

    height, width, _ = img.shape

    left_line, right_line, up_line, down_line = line_detection(img)

    # View lines
    cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 2)   #green
    cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 2)     #yellow
    cv2.line(img, (down_line[0], down_line[1]), (down_line[2], down_line[3]), (255, 0, 255), 2)     #magenta
    cv2.line(img, (up_line[0], up_line[1]), (up_line[2], up_line[3]), (255, 0, 0), 2)               #blue

    point_lu, point_ru, point_rd, point_ld = point_detection(left_line, right_line, up_line, down_line)

    # View point
    #cv2.circle(img, point_rd, 6, (0, 0, 255), -1)
    #cv2.circle(img, point_ru, 6, (0, 0, 255), -1)
    #cv2.circle(img, point_ld, 6, (0, 0, 255), -1)
    #v2.circle(img, point_lu, 6, (0, 0, 255), -1)

    # TODO: remove magic
    # create transformation point
    new_point_ld = point_lu[0], height - magic_indent
    new_point_rd = point_ru[0], height - magic_indent

    # View new point
    # cv2.circle(img, new_point_ld, 6, (255, 0, 255), -1)
    # cv2.circle(img, new_point_rd, 6, (255, 0, 255), -1)

    # Image homography
    img_square_corners = np.float32([point_ru, point_lu, point_ld, point_rd])
    img_quad_corners = np.float32([point_ru, point_lu, new_point_ld, new_point_rd])
    h, mask = cv2.findHomography(img_square_corners, img_quad_corners)



    bird_view = cv2.warpPerspective(img, h, (width, height))

    size_2 = (height * 2, width * 2)
    resize = cv2.resize(bird_view, size_2, interpolation=cv2.INTER_AREA)
    cropped_image = resize[size_2[0]//2:size_2[0]*2, size_2[1]//20:size_2[1]*2]

    # View image
    #cv2.imshow('img', cv2.resize(img, (1920, 1080)))
    #v2.imshow('bird_view', cv2.resize(bird_view, (1920, 1080)))
    cv2.imshow('cropped_image', cv2.resize(cropped_image, (1920, 1080)))
    return bird_view

##not working
def matching_images(camera_img, main_img) :

    #https://habr.com/ru/post/516116/
    #Scale-Invariant Feature Transform creation
    sift = cv2.SIFT_create()

    kp1, descriptors1 = sift.detectAndCompute(camera_img, None)
    kp2, descriptors2 = sift.detectAndCompute(main_img, None)
    img = cv2.drawKeypoints(camera_img, kp1, camera_img)
    img1 = cv2.drawKeypoints(main_img, kp2, main_img)
    #cv2.imshow('camera_img', cv2.resize(camera_img, (1920, 1080)))
    #cv2.imshow('main_img', cv2.resize(img1, (1080, 1080)))

    index_params = dict(algorithm=2, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)




    result = camera_img
    return result


if __name__ == '__main__':

    #img = cv2.imread('resources/dataset/BirdView/001---changzhou/north_1.jpg')
    img = cv2.imread('resources/dataset/BirdView/011---taiyuan/east_1.jpg')
    bird_view = image_homography(img)


   #main_image = cv2.imread('resources/dataset/BirdView/001---changzhou/LongJin_3cm.jpg')
   #main_img = cv2.resize(main_image, (1920, 1920))
   #match = matching_images(bird_view, main_img)

    #cv2.imwrite('../../outt/linesDetected.jpg', img)
    cv2.imwrite('outt/homography.jpg', bird_view)

    cv2.waitKey()