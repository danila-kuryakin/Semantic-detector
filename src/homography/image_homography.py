import cv2
import numpy
import numpy as np
import magicConstans as Constants
import lineOperations
import neural_network.segmentation as nw


def line_detection(img):
    height, width, _ = img.shape

    # gray scale and filtering
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (Constants.gaussianBlurKernel, Constants.gaussianBlurKernel), 0)

    # adaptive thresholding
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th = cv2.medianBlur(th, Constants.medianBlurKernel)
    # find lines
    canny = cv2.Canny(th, 0, 255, None, 3)

    # Line detection
    return cv2.HoughLinesP(canny, 1, np.pi/180, 200, minLineLength=height-height//4, maxLineGap=height//2)


def line_classification(img, lines):
    height, width, _ = img.shape

    start_left_line = [width, 0, width, height]
    left_line = start_left_line
    start_right_line = [0, 0, 0, height]
    right_line = start_right_line

    start_down_line = [0, 0, width, 0]
    down_line = start_down_line
    start_up_line = [0, height, width, height]
    up_line = start_up_line

    vertical_line = []

    for line in lines:

        x1, y1, x2, y2 = line[0]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 127, 127, 255), 2)
        # Classification vertical lines
        if y2 > height-height//4 and y1 < height//4:
            # View lines
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 127), 2)
            vertical_line.append([x1, y1, x2, y2])

        if y1 > height-height//4 and y2 < height//4:
            # View lines
            # cv2.line(img, (x2, y2), (x1, y1), (0, 127, 0), 2)
            vertical_line.append([x2, y2, x1, y1])

        # Classification horizontal lines
        if y1 > height//2 and y2 > height//2:
            [down_x1, down_y1, down_x2, down_y2] = down_line
            # View lines
            # cv2.line(img, (x1, y1), (x2, y2), (127, 0, 0), 1)
            if abs(y1 - y2) < height // 20:
                if y1 > down_y1 and y2 > down_y2:
                    down_line = [x1, y1, x2, y2]
                    continue

        if y1 < height//2 and y2 < height//2 and y1 > height//9 and y2 > height//9:
            [up_x1, up_y1, up_x2, up_y2] = up_line
            # View lines
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            if abs(y1-y2) < height//20:
                if y1 < up_y1 and y2 < up_y2:
                    up_line = [x1, y1, x2, y2]
                    continue

    # View lines
    # [x1, y1, x2, y2] = down_line
    # cv2.line(img, (x1, y1), (x2, y2), (127, 255, 0), 2)

    # View lines
    # [x1, y1, x2, y2] = up_line
    # cv2.line(img, (x1, y1), (x2, y2), (127, 0, 255), 2)

    for line in vertical_line:
        if line[2] > right_line[2]:
            right_line = line
        if line[2] < left_line[2]:
            left_line = line

    if right_line == start_right_line:
        print('right line not found')
    if left_line == start_left_line:
        print('left line not found')
    if down_line == start_down_line:
        print('down line not found')
    return left_line, right_line, up_line, down_line


def point_detection(left_line, right_line, up_line, down_line):
    # Calculation of the equation of a straight line
    right_line_equation = lineOperations.find_line_equation(right_line)
    left_line_equation = lineOperations.find_line_equation(left_line)
    down_line_equation = lineOperations.find_line_equation(down_line)
    up_line_equation = lineOperations.find_line_equation(up_line)

    # Calculation point
    point_rd = lineOperations.find_intersection_point(right_line_equation, down_line_equation)
    point_ru = lineOperations.find_intersection_point(right_line_equation, up_line_equation)
    point_ld = lineOperations.find_intersection_point(left_line_equation, down_line_equation)
    point_lu = lineOperations.find_intersection_point(left_line_equation, up_line_equation)

    return point_lu, point_ru, point_rd, point_ld


def homography(img, objects):
    height, width, color = img.shape
    # TODO: remove magic
    magic_indent = 150

    lines = line_detection(img)
    left_line, right_line, up_line, down_line = line_classification(img, lines)

    # View lines
    # cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 2)
    # cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 2)
    # cv2.line(img, (down_line[0], down_line[1]), (down_line[2], down_line[3]), (255, 0, 05), 2)
    # cv2.line(img, (up_line[0], up_line[1]), (up_line[2], up_line[3]), (127, 0, 255), 2)

    point_lu, point_ru, point_rd, point_ld = point_detection(left_line, right_line, up_line, down_line)

    # View point
    # cv2.circle(img, point_rd, 6, (0, 0, 255), -1)
    # cv2.circle(img, point_ru, 6, (0, 0, 255), -1)
    # cv2.circle(img, point_ld, 6, (0, 0, 255), -1)
    # cv2.circle(img, point_lu, 6, (0, 0, 255), -1)

    new_point_ld = point_lu[0], height - magic_indent
    new_point_rd = point_ru[0], height - magic_indent

    # View new point
    # cv2.circle(img, new_point_ld, 6, (255, 0, 255), -1)
    # cv2.circle(img, new_point_rd, 6, (255, 0, 255), -1)

    # Image homography
    img_square_corners = np.float32([point_ru, point_lu, point_ld, point_rd])
    img_quad_corners = np.float32([point_ru, point_lu, new_point_ld, new_point_rd])
    h, mask = cv2.findHomography(img_square_corners, img_quad_corners)

    warp_objects = []
    for element in objects:
        new_element = dict()
        point1 = (element['xmin'], element['ymin'])
        warp_point1 = lineOperations.warp_point(point1, h)

        point2 = (element['xmax'], element['ymin'])
        warp_point2 = lineOperations.warp_point(point2, h)

        point3 = (element['xmax'], element['ymax'])
        warp_point3 = lineOperations.warp_point(point3, h)

        point4 = (element['xmin'], element['ymax'])
        warp_point4 = lineOperations.warp_point(point4, h)

        new_element['p1'] = warp_point1
        new_element['p2'] = warp_point2
        new_element['p3'] = warp_point3
        new_element['p4'] = warp_point4
        new_element['confidence'] = element['confidence']
        new_element['class'] = element['class']
        new_element['name'] = element['name']
        warp_objects.append(new_element)


    bird_view = cv2.warpPerspective(img, h, (width, height))

    return bird_view, img, warp_objects


def start(path, name):
    # full_path = path + '/' + name
    full_path = path + '/' + name + '.jpg'

    img = cv2.imread(full_path)
    copy_img = numpy.array(img)
    try:
        detected = nw.detect(copy_img, True)
        print(detected)

        bird_view, img_out, detected = homography(img, detected)
        print(detected)

        bird_view = nw.polygon_selection(bird_view, detected)
        # View image
        # cv2.imshow('ld_' + name, cv2.resize(img_out, (720, 480)))
        cv2.imwrite('../../out/ld_' + name, img_out)

        cv2.imshow('h_' + name, cv2.resize(bird_view, (1000, 700)))
        cv2.imwrite('../../out/h_' + name, bird_view)

    except Exception as e:
        print('Error: ', name)
        print(e)


if __name__ == '__main__':

    path = '../resources/dataset/BirdView/001---changzhou'

    start(path, 'east_1')
    # start(path, 'east_2')
    start(path, 'north_1')
    # # start(path, 'east_2')
    start(path, 'south_1')
    # start(path, 'south_2')
    start(path, 'west_1')
    # start(path, 'west_2')

    cv2.waitKey()