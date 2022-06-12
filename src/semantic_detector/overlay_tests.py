import cv2
import numpy as np
import imutils

DB_show_square_lines = 1
DB_show_img = 0
DB_merge_img = 0

# square_coords
square_size = 601
sq_X_0 = 659
sq_X_1 = sq_X_0 + square_size
sq_Y_0 = 239
sq_Y_1 = sq_Y_0 + square_size


# sq_cd_UP_LEFT = (sq_X_0, sq_Y_0)
# sq_cd_UP_RIGHT = (sq_X_1, sq_Y_0)
# sq_cd_DN_LEFT = (sq_X_0, sq_Y_1)
# sq_cd_DN_RIGHT = (sq_X_1, sq_Y_1)

# рисуем недостающие зоны
def image_bg(img_ver, img_N, img_S, img_W, img_E):
    mask1 = np.repeat(np.tile(np.linspace(0, 1, img_N.shape[1]), (img_N.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
    mask2 = imutils.rotate(mask1, 180)
    W_E = np.uint8(img_W * mask1 + img_E * mask2) #swap mask1 and mask2 to change start line
    #cv2.imshow('W_E', cv2.resize(W_E, (1280, 720)))

    mask3 = cv2.resize((imutils.rotate_bound(mask2, 90)), (1920,1080))
    mask4 = imutils.rotate_bound(mask3, 180)
    N_S = np.uint8(img_N * mask4 + img_S * mask3) #swap mask3 and mask4 to change start line
    #cv2.imshow('N_S', cv2.resize(N_S, (1280, 720)))
    # cv2.imshow('mask4', cv2.resize(mask4, (1280, 720)))
    # cv2.imshow('mask3', cv2.resize(mask3, (1280, 720)))
    dst = cv2.addWeighted(N_S, 0.5, W_E, 0.5, 0)
    cv2.imshow('dst', cv2.resize(dst, (1280, 720)))
    return dst


# добавляем центр к перекрестку
def image_square(img_4_cut, img_N, img_S, img_W, img_E):
    blk = np.zeros((601, 601, 3), np.uint8)
    mask1 = np.repeat(np.tile(np.linspace(0, 1, blk.shape[1]), (blk.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
    mask2 = imutils.rotate(mask1, 180)
    img_N_sq = img_N[sq_Y_0:sq_Y_1, sq_X_0:sq_X_1]
    img_S_sq = img_S[sq_Y_0:sq_Y_1, sq_X_0:sq_X_1]
    N_S = np.uint8(img_N_sq * mask1 + img_S_sq * mask2)
    # cv2.imshow('mask1', mask1)
    # cv2.imshow('mask2', mask2)
    # cv2.imshow('img1', np.uint8(img_N_sq * mask1))
    # cv2.imshow('img2', np.uint8(img_S_sq * mask2))
    #cv2.imshow('N_S', N_S)
    mask3 = imutils.rotate(mask1, 90)
    mask4 = imutils.rotate(mask2, 90)
    img_W_sq = img_W[sq_Y_0:sq_Y_1, sq_X_0:sq_X_1]
    img_E_sq = img_E[sq_Y_0:sq_Y_1, sq_X_0:sq_X_1]
    W_E = np.uint8(img_W_sq * mask1 + img_E_sq * mask2)
    #cv2.imshow('W_E', W_E)
    dst = cv2.addWeighted(N_S, 0.5, W_E, 0.5, 0)
    # cv2.imshow('dst', dst)
    merged_center = merge_base_images(img_4_cut, dst, "square")
    # cv2.imshow('merged_center', merged_center)
    return merged_center


# наложение двух каринок друг на друга
def merge_base_images(null_img, merging_img, type):
    heigh_n, width_n = null_img.shape[:2]
    heigh_m, width_m, channels = merging_img.shape

    if type == "North":
        start_y = 0
        end_y = sq_Y_0
        start_x = sq_X_0
        end_x = sq_X_1
    elif type == "South":
        start_y = sq_Y_1
        end_y = 1080
        start_x = sq_X_0
        end_x = sq_X_1
    elif type == "West":
        start_y = sq_Y_0
        end_y = sq_Y_1
        start_x = 350  # 350 bad code
        end_x = sq_X_0
    elif type == "East":
        start_y = sq_Y_0
        end_y = sq_Y_1
        start_x = sq_X_1
        end_x = sq_X_1 + 299  # 299 bad code ошибка при изменении коэффициента из-за несоответствия размеров каринок
    elif type == "square":
        print("check the correct")
        start_y = sq_Y_0
        end_y = sq_Y_1
        start_x = sq_X_0
        end_x = sq_X_1

    cut_n = null_img[start_y:end_y, start_x:end_x]  # separate
    cv2.imshow('cut_n', cut_n) if DB_merge_img == 1 else 0

    img2gray = cv2.cvtColor(merging_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 30, 255, cv2.THRESH_BINARY_INV)  # | cv2.THRESH_OTSU
    cv2.imshow('mask', mask) if DB_merge_img == 1 else 0
    # blurred = cv2.GaussianBlur(mask, (7, 7), 0)
    # cv2.imshow('blurred', blurred) if DB_merge_img == 1 else 0
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mask_inv', mask_inv) if DB_merge_img == 1 else 0

    img1_bg = cv2.bitwise_and(cut_n, cut_n, mask=mask)
    cv2.imshow('img1_bg', img1_bg) if DB_merge_img == 1 else 0
    img2_fg = cv2.bitwise_and(merging_img, merging_img, mask=mask_inv)
    cv2.imshow('img2_fg', img2_fg) if DB_merge_img == 1 else 0

    dst = cv2.add(img1_bg, img2_fg)
    cv2.imshow('dst', dst) if DB_merge_img == 1 else 0
    # blur = cv2.GaussianBlur(dst, (5, 5), cv2.BORDER_TRANSPARENT)
    # dst1 = cv2.addWeighted(blur, 0.5, dst, 0.5, 0)
    # cv2.imshow('dst1', dst1) if DB_merge_img == 1 else 0
    null_img[start_y:end_y, start_x:end_x] = dst  # separate
    cv2.imshow('null_img', null_img) if DB_merge_img == 1 else 0
    return null_img


'''
Отче наш, сущий на небесах! Да святится имя Твоё; да приидет Царствие Твоё; 
да будет воля Твоя и на земле, как на небе; хлеб наш насущный дай нам на сей день; 
и прости нам долги наши, как и мы прощаем должникам нашим; и не введи нас в искушение, 
о избавь нас от лукавого. Ибо Твоё есть Царство и сила и слава во веки. Аминь.
'''


# наложение 4 кусочков изображения с зеброй и линиями
def image_slitching(img_null, img_N, img_S, img_W, img_E):


    # creating North part
    img_N_cut = img_N[0:sq_Y_0, sq_X_0:sq_X_1]
    cv2.imshow('img_N_cut', img_N_cut) if DB_show_img == 1 else 0
    North_null = merge_base_images(img_null, img_N_cut, "North")

    # creating South part
    img_S_cut = img_S[sq_Y_1:1080, sq_X_0:sq_X_1]
    cv2.imshow('img_S_cut', img_S_cut) if DB_show_img == 1 else 0
    South_null = merge_base_images(North_null, img_S_cut, "South")

    # creating West part
    img_W_cut = img_W[sq_Y_0:sq_Y_1, 350:sq_X_0]  # 350 bad code
    cv2.imshow('img_W_cut', img_W_cut) if DB_show_img == 1 else 0
    West_null = merge_base_images(South_null, img_W_cut, "West")

    # creating East part
    img_E_cut = img_E[sq_Y_0:sq_Y_1, sq_X_1:sq_X_1 + 299]  # 299 bad code
    cv2.imshow('img_E_cut', img_E_cut) if DB_show_img == 1 else 0

    East_null = merge_base_images(West_null, img_E_cut, "East")
    # cv2.imshow('East_null', East_null)
    cv2.rectangle(img_null, (sq_X_0, sq_Y_0), (sq_X_1, sq_Y_1), (255, 255, 0), 2) if DB_show_square_lines == 1 else 0
    return East_null


if __name__ == '__main__':

    # при запихивании в функцию необходимо передавать параметр стороны SIDE = S N W E и имя папки для датасета для корректного поворота и сохранения ( а определять такой параметр при загрузке изображения)
    DATASET = 'Data_5'


    img_null = cv2.imread('../../out/null.jpg')
    img_N = cv2.imread('../../out/{}/STEPS/Step_2/North_1080.jpg'.format(DATASET))
    img_S = cv2.imread('../../out/{}/STEPS/Step_2/South_1080.jpg'.format(DATASET))
    img_W = cv2.imread('../../out/{}/STEPS/Step_2/West_1080.jpg'.format(DATASET))
    img_E = cv2.imread('../../out/{}/STEPS/Step_2/East_1080.jpg'.format(DATASET))




    bg_merge = image_bg(img_null, img_N, img_S, img_W, img_E)

    img_4_cut = image_slitching(bg_merge, img_N, img_S, img_W, img_E)
    img_center = image_square(img_4_cut, img_N, img_S, img_W, img_E)

    cv2.imshow('img_center', cv2.resize(img_center, (1280, 720)))

    cv2.waitKey()

#NB
# need to change default bg color to gray for improve color overlaying
#
# центр можно сделать лучще если выбирать не усреднение с каждой стороны а усреднение
# половины центра с каждой стороны ведущего изображения(н/р со стороны слева брать левый верх градиентом,
# сверху правую вертикальную половину градиенгтом и тд