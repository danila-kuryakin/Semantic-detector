import cv2
import numpy as np
import pickle
import statistics
from Step_1_image_homography_v_2 import point_detection

DB_ALL_lines = 0 # показать все линии
DB_step_1_Interest_lines = 0 # показать линии интереса (для каждой стороны)
DB_step_1_Math_lines = 0 #показать ближайшую линию, линии, попадающие в выборку и усредненную линию

Scale_factor = 4    # Для Даниных 1 def = 4
Preview_scale = 1.8 # Для Даниных 1.2 def = 1.8

# при запихивании в функцию необходимо передавать параметр стороны SIDE = S N W E и имя папки для датасета для корректного поворота и сохранения ( а определять такой параметр при загрузке изображения)
SIDE = 'East_1'
DATASET = 'Data_1'

def expand_line(x1, y1, x2, y2, size) :
    if x1==x2 :
        x2+=1
    if y1==y2 :
        y2+=1
    coefficients = np.polyfit([x1, x2], [y1, y2], 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, size, 2)
    y_axis = polynomial(x_axis)
    return round(x_axis[0]),round(y_axis[0]),round(x_axis[1]),round(y_axis[1])

### Blur and edge detection
img_source = cv2.imread('../../out/{}/STEPS/Step_1/{}_1080.jpg'.format(DATASET, SIDE))

height, width, _ = img_source.shape
img = cv2.resize(img_source, (width//Scale_factor, height//Scale_factor))

img_filtre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_filtre = cv2.medianBlur(img_filtre,7)
th3 = cv2.adaptiveThreshold(img_filtre,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

cv2.imwrite('../../out/{}/STEPS/Step_2/_{}_adapThrsh.jpg'.format(DATASET, SIDE), cv2.resize(th3, (width//Scale_factor, height//Scale_factor)))
img_filtre = cv2.imread('../../out/{}/STEPS/Step_2/_{}_adapThrsh.jpg'.format(DATASET, SIDE))
height, width, _ = img_filtre.shape

gray = cv2.cvtColor(img_filtre, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,3)
edged = cv2.Canny(gray, 1, 300)

#cv2.imshow('edged', cv2.resize(edged, (width//Prewiew_scale, height//Prewiew_scale)))
cv2.imwrite('../../out/{}/STEPS/Step_2/_{}_Canny.jpg'.format(DATASET, SIDE), cv2.resize(edged, (width, height)))

# получаем центр изображения из координат линий из гомографии (шаг 1)
variables_file = open('../../out/{}/STEPS/Step_1/{}_img_quad_corners.txt'.format(DATASET, SIDE), "rb")
img_quad_corners = pickle.load(variables_file)
variables_file.close()
img_quad_corners = img_quad_corners * 8
# Для Даниных нужно вписать средние вручную исходя из изображения н/р: center_height, center_width = 557, 989
center_height, center_width = round((abs(img_quad_corners[1][1]-img_quad_corners[2][1]))//2+img_quad_corners[1][1])//Scale_factor, round(abs(img_quad_corners[1][0]))//Scale_factor ### Нужно умнождить на Scale_factor если работать с большим размером
cv2.line(img, (center_width, 0),  (center_width, height), (20, 20, 20), 1) if DB_step_1_Interest_lines == 1 else 0   # vertical center
cv2.line(img, (0, center_height), (width, center_height), (20, 20, 20), 1) if DB_step_1_Interest_lines == 1 else 0  # horizontal center

#[x1, y1, x2, y2]
left_line = [0,0, 0, height]
right_line = [width, 0, width, height]
up_line = [0, 0, width, 0]
down_line = [0, height, width, height]

LINE_SKEW_L_R = 60              # def = 40  предел наклона линий cлева и справа
LINE_SKEW_U_D = 50              # def = 10  предел наклона линий сверху и снизу
MIN_SIZE_NEAREST_LINE_L_R = 200  # def = 97  минимальный размер ближайшей к центру линии cлева и справа (чтобы алгорим не сломался и не выбрал короткую линию как ближайшую к центру
MIN_SIZE_NEAREST_LINE_U = 10   # def = 570  минимальный размер ближайшей к центру линии сверху (чтобы алгорим не сломался и не выбрал короткую линию как ближайшую к центру
MIN_SIZE_NEAREST_LINE_D = 400   # def = 570  минимальный размер ближайшей к центру линии снизу (чтобы алгорим не сломался и не выбрал короткую линию как ближайшую к центру
NEARES_LINE_X_L_R = 0.1          # def = 25  порог выбора ближайших линий, работает только по X cлева и справа
NEARES_LINE_X_U_D = 20         # def = 25  порог выбора ближайших линий, работает только по X сверху и снизу
DIV_LINE_Y = 30                # def = 1.5 фильтр коротких линий по Y, реализовано в условии: (abs(y1-y2) > (abs(left_line[1]-left_line[3]))//DIV_LINE_Y))
# А еще иногда линия может находиться на том же расстоянии что и крайняя и не выделяться алгоритмо(тк используется ">" а не ">="

### Line detection
lines = cv2.HoughLinesP(edged, 2, np.pi / 180, 100, minLineLength=35, maxLineGap=30)# def minLineLength = 10, optimal = 60
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0)) if DB_ALL_lines == 1 else 0


# левая
for line in lines:

    x1, y1, x2, y2 = line[0]

    # Находим линии интереса слева
    if ((y1 > center_height and y2 < center_height and abs(x1-x2) < LINE_SKEW_L_R) or (y2 > center_height and y1 < center_height and abs(x1-x2) < LINE_SKEW_L_R)) and ((x1 < center_width and x1 > 0) or (x2 < center_width and x2 > 0)):

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) if DB_step_1_Interest_lines == 1 else 0

        #находим ближайшую к центру изображения линию слева
        # а еще иногда он выбирает не то условие из ((x2>left_line[2]) or (x1>left_line[0])) тк не оптимальная точка ближе неоптимальной
        if ((x2>left_line[2]) or (x1>left_line[0])) and (MIN_SIZE_NEAREST_LINE_L_R < abs(y1 - y2)) :
            left_line = [x1, y1, x2, y2]
            x1_mean,y1_mean,x2_mean,y2_mean = x1, y1, x2, y2

        #находим все линии относительно ближайшей попадающей в выборку заданную выше( NEARES_LINE_X = 25 ) и усредняем
        # а еще я убрал >0 в ((NEARES_LINE_X_L_R > abs(x1-left_line[0]) > 0  )
        if ((NEARES_LINE_X_L_R > abs(x1-left_line[0])  ) or ((NEARES_LINE_X_L_R > abs(x2-left_line[2]) ))) and ( abs(y1-y2) > (abs(left_line[1]-left_line[3]))//DIV_LINE_Y):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
            x1_mean,y1_mean,x2_mean,y2_mean = statistics.mean([x1, x1_mean]),statistics.mean([y1, y1_mean]),statistics.mean([x2, x2_mean]),statistics.mean([y2, y2_mean])

# ближайшая линия
x1, y1, x2, y2 = left_line[0], left_line[1], left_line[2], left_line[3]
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3) if DB_step_1_Math_lines == 1 else 0
### достроенная линия
x1, y1, x2, y2 = expand_line(x1, y1, x2, y2, height)
left_line_F = [x1, y1, x2, y2]
cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0

# усредненная линия
x1, y1, x2, y2 = x1_mean,left_line[1],x2_mean,left_line[3]
cv2.line(img, (x1, y1), (x2, y2), (0, 256, 0), 2) if DB_step_1_Math_lines == 1 else 0
# перенесенная усредненная линия
if (left_line[0]>left_line[2]) :
    cv2.line(img, (left_line[0], left_line[1]),((left_line[0] - x1_mean) + x2_mean, left_line[3]), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
elif (left_line[2]>=left_line[0]) :
    cv2.line(img, ((left_line[2]-x2_mean) + x1_mean, left_line[1]),(left_line[2], left_line[3]), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0

# правая
for line in lines:

    x1, y1, x2, y2 = line[0]

    # Находим линии интереса справа
    if ((y1 > center_height and y2 < center_height and abs(x1-x2) < LINE_SKEW_L_R) or (y2 > center_height and y1 < center_height and abs(x1-x2) < LINE_SKEW_L_R)) and ((x1 > center_width and x1 < width) or (x2 > center_width and x2 < width)):

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) if DB_step_1_Interest_lines == 1 else 0

        #находим ближайшую к центру изображения линию справа
        if ((x1<right_line[0]) or (x2<right_line[2])) and (MIN_SIZE_NEAREST_LINE_L_R < abs(y1 - y2)) :
            right_line = [x1, y1, x2, y2]
            x1_mean,y1_mean,x2_mean,y2_mean = x1, y1, x2, y2

        #находим все линии относительно ближайшей попадающей в выборку заданную выше( NEARES_LINE_X = 25 ) и усредняем
        if ((NEARES_LINE_X_L_R > abs(x1-right_line[0]) > 0) or ((NEARES_LINE_X_L_R > abs(x2-right_line[2]) > 0))) and ( abs(y1-y2) > (abs(right_line[1]-right_line[3]))//DIV_LINE_Y):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
            x1_mean,y1_mean,x2_mean,y2_mean = statistics.mean([x1, x1_mean]),statistics.mean([y1, y1_mean]),statistics.mean([x2, x2_mean]),statistics.mean([y2, y2_mean])

# ближайшая линия
x1, y1, x2, y2 = right_line[0], right_line[1], right_line[2], right_line[3]
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3) if DB_step_1_Math_lines == 1 else 0
### достроенная линия
x1, y1, x2, y2 = expand_line(x1, y1, x2, y2, height)
right_line_F = [x1, y1, x2, y2]
cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
'''
# усредненная линия
x1, y1, x2, y2 = x1_mean,right_line[1],x2_mean,right_line[3]
cv2.line(img, (x1, y1), (x2, y2), (0, 256, 0), 2) if DB_step_1_Math_lines == 1 else 0
# перенесенная усредненная линия
if (right_line[0]>right_line[2]) :
    cv2.line(img, (right_line[0], right_line[1]),((right_line[0] - x1_mean) + x2_mean, right_line[3]), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
elif (right_line[2]>=right_line[0]) :
    cv2.line(img, ((right_line[2]-x2_mean) + x1_mean, right_line[1]),(right_line[2], right_line[3]), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
'''


# внизу
for line in lines:

    x1, y1, x2, y2 = line[0]

    # Находим линии интереса внизу
    if ((x1 > center_width and x2 < center_width and abs(y1-y2) < LINE_SKEW_U_D) or (x2 > center_width and x1 < center_width and abs(y1-y2) < LINE_SKEW_U_D)) and ((y1 > center_height and y1 < height) or (y2 > center_height and y2 < height)):

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) if DB_step_1_Interest_lines == 1 else 0

        #находим ближайшую к центру изображения линию внизу
        if ((y1<down_line[1]) or (y2<down_line[3])) and (MIN_SIZE_NEAREST_LINE_D < abs(x1 - x2)) :
            down_line = [x1, y1, x2, y2]
            x1_mean,y1_mean,x2_mean,y2_mean = x1, y1, x2, y2
#заменить 0 -->1; 2-->3
        #находим все линии относительно ближайшей попадающей в выборку заданную выше( NEARES_LINE_X = 25 ) и усредняем
        if ((NEARES_LINE_X_U_D > abs(y1-down_line[1]) > 0) or ((NEARES_LINE_X_U_D > abs(y2-down_line[3]) > 0))) and ( abs(x1-x2) > (abs(down_line[0]-down_line[2]))//DIV_LINE_Y):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
            x1_mean,y1_mean,x2_mean,y2_mean = statistics.mean([x1, x1_mean]),statistics.mean([y1, y1_mean]),statistics.mean([x2, x2_mean]),statistics.mean([y2, y2_mean])

# ближайшая линия
x1, y1, x2, y2 = down_line[0], down_line[1], down_line[2], down_line[3]
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3) if DB_step_1_Math_lines == 1 else 0
### достроенная линия
x1, y1, x2, y2 = expand_line(x1, y1, x2, y2, width)
down_line_F = [x1, y1, x2, y2]
cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
'''
# усредненная линия
x1, y1, x2, y2 = down_line[0],y1_mean,down_line[2],y2_mean
cv2.line(img, (x1, y1), (x2, y2), (0, 256, 0), 2) if DB_step_1_Math_lines == 1 else 0
# перенесенная усредненная линия
if (down_line[1]>down_line[3]) :
    cv2.line(img, (down_line[0], down_line[1]),(down_line[2], (down_line[1] - y1_mean) + y2_mean), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
elif (down_line[3]>=down_line[1]) :
    cv2.line(img, (down_line[0], (down_line[3]-y2_mean) + y1_mean),(down_line[2], down_line[3]), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
'''

# вверху
for line in lines:

    x1, y1, x2, y2 = line[0]

    # Находим линии интереса вверху
    if ((x1 > center_width and x2 < center_width and abs(y1-y2) < LINE_SKEW_U_D) or (x2 > center_width and x1 < center_width and abs(y1-y2) < LINE_SKEW_U_D)) and ((y1 < center_height and y1 > 0) or (y2 < center_height and y2 > 0)):

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) if DB_step_1_Interest_lines == 1 else 0

        #находим ближайшую к центру изображения линию вверху
        if ((y1>up_line[1]) or (y2>up_line[3])) and (MIN_SIZE_NEAREST_LINE_U < abs(x1 - x2)) :
            up_line = [x1, y1, x2, y2]
            x1_mean,y1_mean,x2_mean,y2_mean = x1, y1, x2, y2
#заменить 0 -->1; 2-->3
        #находим все линии относительно ближайшей попадающей в выборку заданную выше( NEARES_LINE_X = 25 ) и усредняем
        if ((NEARES_LINE_X_U_D > abs(y1-up_line[1]) > 0) or ((NEARES_LINE_X_U_D > abs(y2-up_line[3]) > 0))) and ( abs(x1-x2) > (abs(up_line[0]-up_line[2]))//DIV_LINE_Y):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0
            x1_mean,y1_mean,x2_mean,y2_mean = statistics.mean([x1, x1_mean]),statistics.mean([y1, y1_mean]),statistics.mean([x2, x2_mean]),statistics.mean([y2, y2_mean])

# ближайшая линия
x1, y1, x2, y2 = up_line[0], up_line[1], up_line[2], up_line[3]
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3) if DB_step_1_Math_lines == 1 else 0

### достроенная линия
x1, y1, x2, y2 = expand_line(x1, y1, x2, y2, width)
up_line_F = [x1, y1, x2, y2]
cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) if DB_step_1_Math_lines == 1 else 0

# усредненная линия
x1, y1, x2, y2 = up_line[0],y1_mean,up_line[2],y2_mean
cv2.line(img, (x1, y1), (x2, y2), (0, 256, 0), 2) if DB_step_1_Math_lines == 1 else 0
# перенесенная усредненная линия
if (up_line[1]>up_line[3]) :
    cv2.line(img, (up_line[0], up_line[1]),(up_line[2], (up_line[1] - y1_mean) + y2_mean), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
    x1, y1, x2, y2 = up_line[0], up_line[1], up_line[2], ((up_line[1] - y1_mean) + y2_mean)
elif (up_line[3]>=up_line[1]) :
    cv2.line(img, (up_line[0], (up_line[3]-y2_mean) + y1_mean),(up_line[2], up_line[3]), (256, 0, 256),2) if DB_step_1_Math_lines == 1 else 0
    x1, y1, x2, y2 = up_line[0], ((up_line[3]-y2_mean) + y1_mean),up_line[2], up_line[3]




point_lu, point_ru, point_rd, point_ld = point_detection(left_line_F, right_line_F, up_line_F, down_line_F)
cv2.circle(img, point_lu, 6, (255, 0, 0), -1)if DB_step_1_Math_lines == 1 else 0
cv2.circle(img, point_ru, 6, (255, 0, 0), -1)if DB_step_1_Math_lines == 1 else 0
cv2.circle(img, point_rd, 6, (255, 0, 0), -1)if DB_step_1_Math_lines == 1 else 0
cv2.circle(img, point_ld, 6, (255, 0, 0), -1)if DB_step_1_Math_lines == 1 else 0


# square_coords
square_size = 601
sq_X_0 = 659
sq_X_1 = sq_X_0 + square_size
sq_Y_0 = 239
sq_Y_1 = sq_Y_0 + square_size
sq_cd_UP_LEFT = (sq_X_0, sq_Y_0)
sq_cd_UP_RIGHT = (sq_X_1, sq_Y_0)
sq_cd_DN_LEFT = (sq_X_0, sq_Y_1)
sq_cd_DN_RIGHT = (sq_X_1, sq_Y_1)
res_image = np.zeros((1080,1920,3), np.uint8)
img_square_corners = np.float32([point_ru, point_lu, point_ld, point_rd])
# Сторона влияет на порядок точек в img_quad_corners. Стандартный обход: sq_cd_UP_RIGHT, sq_cd_UP_LEFT, sq_cd_DN_LEFT, sq_cd_DN_RIGHT
# Однако, для каждой стороны точки соответственно будут меняться местами для отражения изображения
if SIDE == 'North':
    print('Выбрана North')
    img_quad_corners = np.float32([sq_cd_DN_LEFT, sq_cd_DN_RIGHT, sq_cd_UP_RIGHT, sq_cd_UP_LEFT])
elif SIDE == 'South':
    print ('Выбрана South')
    img_quad_corners = np.float32([sq_cd_UP_RIGHT, sq_cd_UP_LEFT, sq_cd_DN_LEFT, sq_cd_DN_RIGHT])
elif SIDE == 'West':
    print('Выбрана West')
    img_quad_corners = np.float32([sq_cd_DN_RIGHT, sq_cd_UP_RIGHT, sq_cd_UP_LEFT, sq_cd_DN_LEFT])
elif SIDE == 'East':
    print('Выбрана East')
    img_quad_corners = np.float32([sq_cd_UP_LEFT, sq_cd_DN_LEFT, sq_cd_DN_RIGHT, sq_cd_UP_RIGHT])

h, mask = cv2.findHomography(img_square_corners, img_quad_corners)
res_image = cv2.warpPerspective(img, h, (1920, 1080))
cv2.imshow('blank_image',cv2.resize(res_image, (1280, 720)))
cv2.imwrite('../../out/{}/STEPS/Step_2/{}_1080.jpg'.format(DATASET,SIDE), cv2.resize(res_image, (1920, 1080)))


cv2.imshow('Final', cv2.resize(img, (round(width // Preview_scale), round(height // Preview_scale))))

cv2.waitKey(0)
cv2.destroyAllWindows()



