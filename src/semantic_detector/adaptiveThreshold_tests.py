import cv2
import numpy as np
import pickle
import statistics

DB_ALL_lines = 0 # показать все линии
DB_step_1_Interest_lines = 1 # показать линии интереса (для каждой стороны)
DB_step_1_Math_lines = 1 #показать ближайшую линию, линии, попадающие в выборку и усредненную линию

Scale_factor = 4
Preview_scale = 1.8

### Blur and edge detection
img_source = cv2.imread('E:\Semantic-detector\out/5/5_west_HR.jpg')
height, width, _ = img_source.shape
img = cv2.resize(img_source, (width//Scale_factor, height//Scale_factor))

img_filtre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_filtre = cv2.medianBlur(img_filtre,7)
th3 = cv2.adaptiveThreshold(img_filtre,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

cv2.imwrite('../../out/0_adapThrsh.jpg', cv2.resize(th3, (width//Scale_factor, height//Scale_factor)))
img_filtre = cv2.imread('../../out/0_adapThrsh.jpg')
height, width, _ = img_filtre.shape

gray = cv2.cvtColor(img_filtre, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,3)
edged = cv2.Canny(gray, 1, 300)

#cv2.imshow('edged', cv2.resize(edged, (width//Prewiew_scale, height//Prewiew_scale)))
cv2.imwrite('../../out/1_Canny.jpg', cv2.resize(edged, (width//Scale_factor, height//Scale_factor)))





# получаем центр изображения из координат линий из гомографии (шаг 1)
variables_file = open("img_quad_corners.txt", "rb")
img_quad_corners = pickle.load(variables_file)
variables_file.close()
img_quad_corners = img_quad_corners * 8
center_height, center_width = round((abs(img_quad_corners[1][1]-img_quad_corners[2][1]))//2+img_quad_corners[1][1])//Scale_factor, round(abs(img_quad_corners[1][0]))//Scale_factor ### Нужно умнождить на Scale_factor если работать с большим размером


### Line detection
lines = cv2.HoughLinesP(edged, 2, np.pi / 180, 100, minLineLength=10, maxLineGap=30)



#[x1, y1, x2, y2]
left_line = [0,0, 0, height]
mass_left_line = [left_line,left_line] # первая ближайшая к центру, !!!надо реформатировать, оставив только первую
right_line = [center_width, 0, center_width, height]
up_line = [0, center_height, width, center_height]
down_line = [0, center_height, width, center_height]
NEARES_LINE_X = 25 # порог выбора ближайших линий (работает только по X, по Y большое значение)
NEARES_LINE_Y = 20000
delta = 0

cv2.line(img, (center_width+delta, 0),  (center_width+delta, height), (0, 250, 250), 1)   # left_line
cv2.line(img, (center_width-delta, 0),  (center_width-delta, height), (0, 250, 0), 1)     # right_line
cv2.line(img, (0, center_height+delta), (width, center_height+delta), (250, 0, 250), 1)   # up_line
cv2.line(img, (0, center_height-delta), (width, center_height-delta), (0, 0, 250), 1)     # down_line
#cv2.imshow('edged1', cv2.resize(img, (int(1280*1.1), int(720*1.1))))



for line in lines:

    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0)) if DB_ALL_lines == 1 else 0

    # Находим линии интереса слева
    if ((y1 > center_height and y2 < center_height and abs(x1-x2) < 40) or (y2 > center_height and y1 < center_height and abs(x1-x2) < 40)) and ((x1 < center_width and x1 > 0) or (x2 < center_width and x2 > 0)):

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) if DB_step_1_Interest_lines == 1 else 0

        #находим ближайшую к центру изображения линию слева
        if (x1>mass_left_line[0][0]) or (x2>mass_left_line[0][2]) :
            mass_left_line[0] = [x1, y1, x2, y2]
            x1_mean,y1_mean,x2_mean,y2_mean = x1, y1, x2, y2

        #находим все линии относительно ближайшей попадающей в выборку заданную выше( NEARES_LINE_X = 25 ) и усредняем
        if ((NEARES_LINE_X > abs(x1-mass_left_line[0][0]) > 0) or ((NEARES_LINE_X > abs(x2-mass_left_line[0][2]) > 0))) and ((NEARES_LINE_Y > abs(y1-mass_left_line[0][1]) > 0) or ((NEARES_LINE_Y > abs(y2-mass_left_line[0][3]) > 0))):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
            x1_mean,y1_mean,x2_mean,y2_mean = statistics.mean([x1, x1_mean]),statistics.mean([y1, y1_mean]),statistics.mean([x2, x2_mean]),statistics.mean([y2, y2_mean])

# ближайшая линия
x1, y1, x2, y2 = mass_left_line[0][0], mass_left_line[0][1], mass_left_line[0][2], mass_left_line[0][3]
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2) if DB_step_1_Math_lines == 1 else 0
# усредненная линия
x1, y1, x2, y2 = x1_mean,mass_left_line[0][1],x2_mean,mass_left_line[0][3]
cv2.line(img, (x1, y1), (x2, y2), (0, 256, 0), 2) if DB_step_1_Math_lines == 1 else 0


# перенесенная усредненная линия
cv2.line(img, (mass_left_line[0][0], mass_left_line[0][1]), ((mass_left_line[0][0] - x1_mean) + x2_mean, mass_left_line[0][3]), (256, 0, 256), 2) if DB_step_1_Math_lines == 1 else 0

cv2.imshow('Final', cv2.resize(img, (round(width // Preview_scale), round(height // Preview_scale))))



cv2.imshow('Final', cv2.resize(img, (round(width//Preview_scale), round(height//Preview_scale))))
cv2.imwrite('../../out/Lines.jpg', cv2.resize(img, (round(width), round(height))))


cv2.waitKey(0)
cv2.destroyAllWindows()



