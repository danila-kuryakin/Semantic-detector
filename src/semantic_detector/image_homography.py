import cv2
import numpy
import numpy as np

# Searches for linear equation variables
# y = kx + b
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


def line_detection(img):
    height, width, _ = img.shape

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', cv2.resize(gray, (400, 300)))

    # Darkness
    darkGrey = cv2.addWeighted(gray, 0.4, np.zeros(gray.shape, gray.dtype), 0.6, 0.0)
    # cv2.imshow("DarkGrey", cv2.resize(darkGrey, (400, 300)))

    # HLS
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # cv2.imshow("imgHLS", cv2.resize(imgHLS, (400, 300)))

    # Yellow highlight
    HLSYellow = cv2.inRange(imgHLS, (10, 100, 60), (40, 255, 255))
    # cv2.imshow("HLSYellow", cv2.resize(HLSYellow, (400, 300)))

    # White highlight
    HLSWhite = cv2.inRange(imgHLS, (0, 180, 0), (255, 255, 255))
    # cv2.imshow("HLSWhite", cv2.resize(HLSWhite, (400, 300)))

    # Create Mask
    mask = cv2.bitwise_or(HLSWhite, HLSYellow)
    finallyMask = cv2.bitwise_and(darkGrey, darkGrey, mask=mask)
    # cv2.imshow("finallyMask", cv2.resize(finallyMask, (400, 300)))

    # GaussianBlur for Mask
    gauss = cv2.GaussianBlur(finallyMask, (7, 7), cv2.BORDER_DEFAULT)
    # cv2.imshow("GaussianBlur", cv2.resize(gauss, (400, 300)))

    # Canny for GaussianBlur
    edges = cv2.Canny(gauss, 50, 150, apertureSize=7)
    # cv2.imshow('Canny', cv2.resize(edges, (400, 300)))

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=height-height//2, maxLineGap=height)

    start_left_line = [width, 0, width, height]
    left_line = start_left_line
    start_right_line = [0, 0, 0, height]
    right_line = start_right_line

    start_down_line = [0, height, width, height]
    down_line = start_down_line

    vertical_line = []

    for line in lines:

        x1, y1, x2, y2 = line[0]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 127, 127), 2)
        # Classification vertical lines
        if y2 > height-height//4 and y1 < height//4:
            # View lines
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 127), 2)
            vertical_line.append([x1, y1, x2, y2])

        if y1 > height-height//4 and y2 < height//4:
            # View lines
            # cv2.line(img, (x2, y2), (x1, y1), (0, 127, 0), 2)
            vertical_line.append([x2, y2, x1, y1])

        # TODO: redo please
        # Classification horizontal lines
        if y1 > height//2 and y2 > height//2:
            # View lines
            # cv2.line(img, (x1, y1), (x2, y2), (127, 0, 0), 2)
            if y1 < down_line[1] and y2 < down_line[3]:
                down_line = [x1, y1, x2, y2]
            continue

    for line in vertical_line:
        if line[2] > right_line[2]:
            right_line = line
        if line[2] < left_line[2]:
            left_line = line


    # TODO: bad idea
    # Find up line
    up_line = (down_line[0], down_line[1] - height // 3, down_line[2], down_line[3] - height // 3)

    if right_line == start_right_line:
        print('right line not found')
    if left_line == start_left_line:
        print('left line not found')
    if down_line == start_down_line:
        print('down line not found')
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

    height, width, _ = img.shape
    # TODO: remove magic
    magic_indent = 150

    left_line, right_line, up_line, down_line = line_detection(img)

    # View lines
    cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 2)
    cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 2)
    cv2.line(img, (down_line[0], down_line[1]), (down_line[2], down_line[3]), (255, 0, 0), 2)
    cv2.line(img, (up_line[0], up_line[1]), (up_line[2], up_line[3]), (255, 0, 0), 2)

    point_lu, point_ru, point_rd, point_ld = point_detection(left_line, right_line, up_line, down_line)

    # View point
    cv2.circle(img, point_rd, 6, (0, 0, 255), -1)
    cv2.circle(img, point_ru, 6, (0, 0, 255), -1)
    cv2.circle(img, point_ld, 6, (0, 0, 255), -1)
    cv2.circle(img, point_lu, 6, (0, 0, 255), -1)

    # TODO: remove magic
    # create transformation point
    new_point_ld = point_lu[0], height - magic_indent
    new_point_rd = point_ru[0], height - magic_indent

    # View new point
    cv2.circle(img, new_point_ld, 6, (255, 0, 255), -1)
    cv2.circle(img, new_point_rd, 6, (255, 0, 255), -1)

    # Image homography
    img_square_corners = np.float32([point_ru, point_lu, point_ld, point_rd])
    img_quad_corners = np.float32([point_ru, point_lu, new_point_ld, new_point_rd])
    h, mask = cv2.findHomography(img_square_corners, img_quad_corners)
    bird_view = cv2.warpPerspective(img, h, (width, height))

    return bird_view


def start(path, name):
    full_path = path + '/' + name

    img = cv2.imread(full_path)
    try:
        bird_view = image_homography(img)

        # View image
        cv2.imshow('ld_' + name, cv2.resize(img, (400, 300)))
        cv2.imwrite('../../out/ld_' + name, img)

        cv2.imshow('h_' + name, cv2.resize(bird_view, (400, 300)))
        cv2.imwrite('../../out/h_' + name, bird_view)
    except Exception as e:
        print('Error: ', name)
        print(e)


if __name__ == '__main__':

    path = '../resources/dataset/BirdView/001---changzhou'

    start(path, 'west_1.jpg')
    # start(path, 'east_1.jpg')
    # start(path, 'south_1.jpg')
    # start(path, 'north_1.jpg')

    cv2.waitKey()