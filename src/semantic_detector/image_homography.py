import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import math

# Searches for linear equation variables
# y = kx + b
from cv2 import CV_32F, flann


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
    cv2.imshow('edges', cv2.resize(edges, (400, 300)))

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
    up_line = (down_line[0], down_line[1]-20 - height // 2, down_line[2], down_line[3]-20 - height // 2)# start_width start_heigh end_width end_heigh

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

    #Line Start coordinates
    LEFT_ls = [200,20,100,430]
    LEFT_center = [((math.fabs(LEFT_ls[0]-LEFT_ls[2]))/2), (math.fabs(LEFT_ls[1]-LEFT_ls[3]))/2]
    RIGHT_ls = [600,20,700,430]
    RIGHT_center = [((math.fabs(RIGHT_ls[0]-RIGHT_ls[2]))/2), (math.fabs(RIGHT_ls[1]-RIGHT_ls[3]))/2]
    UP_ls = [20,50,780,50]
    UP_center = [((math.fabs(UP_ls[0]-UP_ls[2]))/2), (math.fabs(UP_ls[1]-UP_ls[3]))/2]
    DOWN_ls = [20,350,780,350]
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

    info_text = tk.Label(win, text = 'Change position and rotation rulers to select perspective distortion and click ???',
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
    LEFT_line = workspace.create_line(LEFT_ls, fill="#0AB6FF", width=5,
                          activefill = '#FF00D4',
                          )
    RIGHT_line = workspace.create_line(RIGHT_ls, fill="#26ff1f", width=5,
                          activefill = '#FF00D4'
                          )
    UP_line = workspace.create_line(UP_ls, fill="#F8FF24", width=5,
                          activefill = '#FF00D4'
                          )
    DOWN_line = workspace.create_line(DOWN_ls, fill="#F56A00", width=5,
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
            rotation_speed=0.5
        elif event.char == "2":
            rotation_speed=3
        elif event.char == "3":
            rotation_speed=7
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
    # TODO: remove magic
    magic_indent = 150

    height, width, _ = img.shape

    left_line, right_line, up_line, down_line = Hand_line_detection(img) #line_detection(img)

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
    cv2.imshow('img', cv2.resize(img, (800, 600)))
    cv2.imshow('bird_view', cv2.resize(bird_view, (1920, 1080)))
    #cv2.imshow('cropped_image', cv2.resize(cropped_image, (1920, 1080)))
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

    #img = cv2.imread('../resources/dataset/BirdView/010---rongguixinma/south_1.jpg')
    img = cv2.imread('../resources/dataset/BirdView/013---yancheng/west_1.jpg')
    bird_view = image_homography(img)


   #main_image = cv2.imread('../resources/dataset/BirdView/001---changzhou/LongJin_3cm.jpg')
   #main_img = cv2.resize(main_image, (1920, 1920))
   #match = matching_images(bird_view, main_img)

    #cv2.imwrite('../../Semantic-detector/out/linesDetected.jpg', img)
    cv2.imwrite('../../out/homography.jpg', bird_view)

    cv2.waitKey()