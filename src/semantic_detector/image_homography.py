import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import math


#Ability to show images of all stages of the program 0/1
DB_show_images = 0
DB_show_ALG_lines = 0
DB_show_ALG_lines_ALL = 1
DB_show_FIN_lines_points = 1


def find_line_equation(line):
    x1, y1, x2, y2 = line
    if ((x1 - x2))!=0:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k * x2
        return {'k': k, 'b': b}
    else:
        return {'k': 0, 'b': 0}

# Looking for the point where the lines intersect
def find_intersection_point(line_a, line_b):
    x = int((line_b['b'] - line_a['b'])/(line_a['k'] - line_b['k']))
    y = int(line_b['k'] * x + line_b['b'])
    return x, y

def apply_threshold(filtered):
    ret, thresh = cv2.threshold(filtered, 254, 255, cv2.THRESH_OTSU)
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    return thresh

def line_detection(img):

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Darkness
    darkGrey = cv2.addWeighted(gray, 0.4, np.zeros(gray.shape, gray.dtype), 0.6, 0.0)
    # HLS
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Yellow highlight
    HLSYellow = cv2.inRange(imgHLS, (10, 60, 60), (40, 210, 255))
    # White highlight
    HLSWhite = cv2.inRange(imgHLS, (0, 180, 0), (255, 255, 255))
    # Create Mask
    mask = cv2.bitwise_or(HLSWhite, HLSYellow)
    # Create GrayMask
    #finallyMask = cv2.bitwise_and(darkGrey, darkGrey, mask=mask)
    # GaussianBlur for Mask
    gauss = cv2.GaussianBlur(mask, (7, 7), cv2.BORDER_DEFAULT)
    # Canny for GaussianBlur
    testEdges = cv2.Canny(gauss, 1, 1, apertureSize=3)

    if DB_show_images == 1 :
        cv2.imshow("Original", cv2.resize(img, (800, 450)))
        cv2.imshow("Gray", cv2.resize(gray, (800, 450)))
        cv2.imshow("DarkGrey", cv2.resize(darkGrey, (800, 450)))
        cv2.imshow("imgHLS", cv2.resize(imgHLS, (800, 450)))
        cv2.imshow("HLSYellow", cv2.resize(HLSYellow, (800, 450)))
        cv2.imshow("HLSWhite", cv2.resize(HLSWhite, (800, 450)))
        cv2.imshow("mask", cv2.resize(mask, (800, 450)))
        cv2.imshow("GaussianBlur", cv2.resize(gauss, (800, 450)))
        cv2.imshow('TestEdges.jpg', cv2.resize(testEdges, (800, 450)))



    # Line detection
    height, width, _ = img.shape
    lines = cv2.HoughLinesP(testEdges, 2, np.pi/180, 250, minLineLength=height-height//5, maxLineGap=height)
    left_line   = [width, 0, width, height]
    right_line  = [0, 0, 0, height]
    up_line     = [0, 0, width, 0]
    down_line   = [0, height, width, height]

    for line in lines:

        x1, y1, x2, y2 = line[0]
        # View all lines
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1) if DB_show_ALG_lines_ALL == 1 else 0

        # Classification vertical lines
        if y2 > height-height//4 and y1 < height//4:
            # View lines
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 150), 3) if DB_show_ALG_lines == 1 else 0

            if x2 > right_line[2]: #x1 not Used
                right_line = [x1, y1, x2, y2]
            continue

        if y1 > height-height//4 and y2 < height//4:
            # View lines
            cv2.line(img, (x2, y2), (x1, y1), (0, 150, 150), 3) if DB_show_ALG_lines == 1 else 0

            if x1 < left_line[2]:
                left_line = [x2, y2, x1, y1]
            continue


        # Classification down line
        if y1 > height - height//2 and y2 > height - height//2: # нужно вынести как переменные масштабирование
            # View lines
            cv2.line(img, (x1, y1), (x2, y2), ( 150, 0,0), 1) if DB_show_ALG_lines == 1 else 0

            skew_error = 100  # ошибка поворота когда линия слишком широкая и с одной стороны видно верх а с другой низ, выбираем среднее
            if y1 < down_line[1] and y2 < down_line[3]:
                down_line = [x1, y1, x2, y2]
            skew_down = math.fabs(down_line[1]-down_line[3])
            if  skew_down < skew_error:
                down_line[1] = int(down_line[1] + skew_down // 2)
                down_line[3] = int(down_line[3] - skew_down // 2)
            continue

        # Classification up line
        if y1 < height // 3.7 and y2 < height // 3.7 : # нужно вынести как переменные масштабирование
            # View lines
            cv2.line(img, (x1, y1), (x2, y2), (150, 0, 150), 1) if DB_show_ALG_lines == 1 else 0

            skew_threshold = 100 #пытаемся найти самую нижнюю линию, если наклон по одной из сторон меньше порогового значения
            if y1 > up_line[1] and y2 > up_line[3]:
                up_line = [x1, y1, x2, y2]
            if y1 > up_line[1] and math.fabs(up_line[1]-up_line[3])<skew_threshold:
                up_line[1]  = y1
            if y2 > up_line[3] and math.fabs(up_line[1]-up_line[3])<skew_threshold:
                up_line[3] = y2
            continue


    # TODO: bad idea
    # Find up line
    #up_line = (down_line[0], down_line[1]+150 - height // 2, down_line[2], down_line[3]+150 - height // 2)# start_width start_heigh end_width end_heigh

    return left_line, right_line, up_line, down_line



def Hand_line_detection(img):
    global zero_angle
    global rotation_speed
    global line_switch
    #window sizes and importing image
    heigh_S = 800
    width_S = 600
    imported_image = img #cv2.imread('north_1.jpg')
    imported_image = cv2.cvtColor(imported_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(imported_image, (800, 450))

    #read saved coordinates
    variables_file = open("variables_file.txt", "rb")
    variables_data = pickle.load(variables_file)
    variables_file.close()

    #Line Start coordinates
    LEFT_ls = variables_data[0] #[200,20,100,830]
    LEFT_center = [((math.fabs(LEFT_ls[0]-LEFT_ls[2]))/2), (math.fabs(LEFT_ls[1]-LEFT_ls[3]))/2]
    RIGHT_ls = variables_data[1] #[600,20,700,830]
    RIGHT_center = [((math.fabs(RIGHT_ls[0]-RIGHT_ls[2]))/2), (math.fabs(RIGHT_ls[1]-RIGHT_ls[3]))/2]
    UP_ls = variables_data[2] #[20,50,780,50]
    UP_center = [((math.fabs(UP_ls[0]-UP_ls[2]))/2), (math.fabs(UP_ls[1]-UP_ls[3]))/2]
    DOWN_ls = variables_data[3] #[-200,350,1280,350]
    DOWN_center = [((math.fabs(DOWN_ls[0]-DOWN_ls[2]))/2), (math.fabs(DOWN_ls[1]-DOWN_ls[3]))/2]

    Line_Starts = [LEFT_ls, RIGHT_ls, UP_ls, DOWN_ls]
    Centers = [LEFT_center, RIGHT_center, UP_center, DOWN_center]
    zero_angle = [280,80,0,180] # start rotating angle

    line_switch =-1 #switching between four lines in functions
    rotation_speed = 7 #default rotation step

    #You can add function, helping you to move start and end points of all lines by using this method
    #workspace.coords(RIGHT_line, 5, 100, 400, 250) where 5, 100, 400, 250 = Start_X, Start_Y, End_X, End_Y
    #To do this, create two rectangles. their centers will be the start and end points of the curve
    #I don' want do this)))

    #Function called by clicking OK button
    def accept_rulers ():
        global variables_all
        left_line = workspace.coords(LINE_name[0])
        right_line = workspace.coords(LINE_name[1])
        up_line = workspace.coords(LINE_name[2])
        down_line = workspace.coords(LINE_name[3])

        variables_all = (left_line, right_line, up_line, down_line)
        variables_file = open("variables_file.txt", "wb")
        pickle.dump(variables_all, variables_file)
        variables_file.close()

        for i in range (4):
            left_line[i] = round(left_line[i] * 2.4)
            right_line[i] = round(right_line[i] * 2.4)
            up_line[i] = round(up_line[i] * 2.4)
            down_line[i] = round(down_line[i] * 2.4)
        win.destroy()
        variables_all = (left_line, right_line, up_line, down_line)
        print(variables_all)



    #Here we have to take line start and end point and pass variables to basics code

    win = tk.Tk()
    win.title('Rulers input')
    win.geometry(f"{heigh_S}x{width_S}+1100+200")
    win.resizable(False,False)
    win.columnconfigure(1, minsize = 200)

    info_text = tk.Label(win, text = 'Change position and rotation rulers to select perspective distortion and click →',
                           font = ('Arial', 12, 'normal'),
                           width = 66,
                           heigh = 2,
                           bg = '#cee4e4'
                           )
    info_hotkeys = tk.Label(win, text = 'Use 1 2 3 keys to control rotation speed',
                           font = ('Arial', 12, 'normal'),
                           width = 66,
                           heigh = 2,
                           bg = '#cee4e4'
                           )
    accept_btn = tk.Button(win, text = 'OK',
                           command = accept_rulers,
                           font = ('Arial', 10, 'normal'),
                           width = 10,
                           heigh = 2,
                           )
    workspace = tk.Canvas(height=800, width=450
                       )

    #Add image in canvas
    im = Image.fromarray(resized_image)
    imgtk = ImageTk.PhotoImage(image=im)
    image = workspace.create_image(0, 0, anchor='nw',image=imgtk)

    #Add lines in canvas
    LEFT_line = workspace.create_line(LEFT_ls, fill="#0AB6FF", width=2,
                          activefill = '#FF00D4',
                          )
    RIGHT_line = workspace.create_line(RIGHT_ls, fill="#26ff1f", width=2,
                          activefill = '#FF00D4'
                          )
    UP_line = workspace.create_line(UP_ls, fill="#F8FF24", width=2,
                          activefill = '#FF00D4'
                          )
    DOWN_line = workspace.create_line(DOWN_ls, fill="#F56A00", width=2,
                          activefill = '#FF00D4'
                          )
    LINE_name = [LEFT_line, RIGHT_line, UP_line, DOWN_line]

    def on_closing():
        if messagebox.askokcancel("Wrong way((", "Reminder! Line coordinates are not transferred when the program is closed. Do you want to quit? "):
            win.destroy()

    def move_ruler(event, arg):

        print(event.x_root, event.y_root)
        mouse_x = workspace.winfo_pointerx()-workspace.winfo_rootx()
        mouse_y = workspace.winfo_pointery()-workspace.winfo_rooty()
        print(mouse_x, mouse_y)
        workspace.moveto(LINE_name[arg], mouse_x-Centers[arg][0]-5, mouse_y-Centers[arg][1]-5)

        workspace.bind("<MouseWheel>", lambda event, num_val = arg : mouse_wheel(event, num_val))
        #workspace.move(LEFT_line, mouse_x, mouse_y)
        #workspace.place(x = mouse_x, y = mouse_y, anchor='center')

    #Rotation
    def Rotate(event,arg):
        #print('rotate')
        angle_in_radians = zero_angle[arg] * math.pi / 180
        line_length = math.hypot(Centers[arg][0], Centers[arg][1])
        start_x = Centers[arg][0] - line_length * math.cos(angle_in_radians)
        start_y = Centers[arg][1] - line_length * math.sin(angle_in_radians)
        end_x = Centers[arg][0] + line_length * math.cos(angle_in_radians)
        end_y = Centers[arg][1] + line_length * math.sin(angle_in_radians)
        workspace.coords(LINE_name[arg], start_x, start_y, end_x, end_y)
        workspace.move(LINE_name[arg], event.x - (Centers[arg][0]) , event.y - (Centers[arg][1]) )
        Line_Starts[arg] = (start_x,start_y,end_x,end_y)
        Centers[arg] = (((math.fabs(Line_Starts[arg][0]-Line_Starts[arg][2]))/2), ((math.fabs(Line_Starts[arg][1]-Line_Starts[arg][3]))/2))

    def mouse_wheel(event, arg):
        global zero_angle
        # respond to Linux or Windows wheel event
        if event.num == 5 or event.delta == -120:
            zero_angle[arg] += rotation_speed
        if event.num == 4 or event.delta == 120:
            zero_angle[arg] -= rotation_speed
        Rotate(event, arg)
        print(zero_angle[arg])

    #0 Left #1 Right #2 Up #3 Down
    def key_switch(event):
        global rotation_speed
        global line_switch

        #Switch rotation speed
        if event.char == "1":
            rotation_speed=0.2
        elif event.char == "2":
            rotation_speed=1
        elif event.char == "3":
            rotation_speed=5
        else:
            return 0
        #print(event.char)
    win.bind("<Key>",key_switch)

    workspace.tag_bind(LINE_name[0], "<B1-Motion>", lambda event, arg = 0 : move_ruler(event, arg))
    workspace.tag_bind(LINE_name[1], "<B1-Motion>", lambda event, arg = 1 : move_ruler(event, arg))
    workspace.tag_bind(LINE_name[2], "<B1-Motion>", lambda event, arg = 2 : move_ruler(event, arg))
    workspace.tag_bind(LINE_name[3], "<B1-Motion>", lambda event, arg = 3 : move_ruler(event, arg))

    #Rendering objects on window
    info_text.grid(row = 0, column = 0)
    accept_btn.grid(row = 0, column = 1, columnspan = 2, stick = 'we')
    info_hotkeys.grid(row = 1, column = 0,columnspan = 2, stick = 'we')
    workspace.grid(row=2,column=0,columnspan = 2, stick = 'we')
    win.protocol("WM_DELETE_WINDOW", on_closing)
    win.mainloop()
    return variables_all

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

    #find line coordinates
    left_line, right_line, up_line, down_line = line_detection(img) #Hand_line_detection(img)
    # find point coordinates
    point_lu, point_ru, point_rd, point_ld = point_detection(left_line, right_line, up_line, down_line)

    ''' 
    #Find left-right lines intersection
    right_line_equation = find_line_equation(right_line)
    left_line_equation = find_line_equation(left_line)
    point_up = find_intersection_point(left_line_equation, right_line_equation)
    cv2.circle(img, point_up, 6, (255, 0, 0), -1)
    '''

    # TODO: remove magic
    height, width, _ = img.shape
    # create transformation point
    new_point_ld = point_lu[0], point_ld[1] #height - magic_indent magic_indent = 100
    new_point_rd = point_ru[0], point_ld[1] #point_rd[1]
    new_point_ru = point_ru[0], point_lu[1]

    # View lines and points and new point
    if DB_show_FIN_lines_points == 1:
        cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)   #green
        cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 5)     #yellow
        cv2.line(img, (down_line[0], down_line[1]), (down_line[2], down_line[3]), (255, 0, 255), 5)     #magenta
        cv2.line(img, (up_line[0], up_line[1]), (up_line[2], up_line[3]), (255, 0, 0), 5)               #blue
        cv2.circle(img, point_rd, 6, (0, 0, 255), -1)
        cv2.circle(img, point_ru, 6, (0, 0, 255), -1)
        cv2.circle(img, point_ld, 6, (0, 0, 255), -1)
        cv2.circle(img, point_lu, 6, (0, 0, 255), -1)
        cv2.circle(img, new_point_ld, 6, (255, 0, 255), -1)
        cv2.circle(img, new_point_rd, 6, (255, 0, 255), -1)

    # homography value 1
    img_square_corners = np.float32([point_ru, point_lu, point_ld, point_rd])

    # Reset crossway width proportions
    mnoj = math.fabs((point_ru[0]-point_lu[0])/(point_rd[0] -point_ld[0]))*0.8
    point_ru, point_lu, new_point_ld, new_point_rd = [new_point_ru[0]*mnoj,new_point_ru[1]],[point_lu[0]*mnoj,point_lu[1]],[new_point_ld[0]*mnoj,new_point_ld[1]],[new_point_rd[0]*mnoj,new_point_rd[1]]

    # homography value 2
    img_quad_corners = np.float32([point_ru, point_lu, new_point_ld, new_point_rd])

    # Image homography
    h, mask = cv2.findHomography(img_square_corners, img_quad_corners)
    bird_view = cv2.warpPerspective(img, h, (width, height))
    cv2.imshow('bird_view', cv2.resize(bird_view, (1000, 563))) if DB_show_images == 1 else 0

    #Crop the image
    c = mnoj/5
    a = int(height*(1-(mnoj-c)))
    b = int(width*(mnoj+c))
    cropped = bird_view[0:a, 0:b]
    cv2.imshow('cropped', cv2.resize(cropped, (b, a)))
    cv2.imwrite('../../out/homography.jpg', cv2.resize(cropped, (b, a), interpolation= cv2.INTER_CUBIC))

    # View image
    cv2.imshow('Null', cv2.resize(img, (1000, 563)))
    return bird_view


if __name__ == '__main__':

    img = cv2.imread('../resources/dataset/BirdView/008---lean/west_1.jpg')
    bird_view = image_homography(img)
    #cv2.imwrite('../../out/homography.jpg', cv2.resize(bird_view, (1920, 1080)))

    cv2.waitKey()