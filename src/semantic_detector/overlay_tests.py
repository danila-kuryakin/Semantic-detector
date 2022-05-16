import cv2

DB_show_square_lines = 1
DB_show_img = 0

# square_coords
sq_X_0 = 659
sq_X_1 = 1260
sq_Y_0 = 239
sq_Y_1 = 840
sq_cd_UP_LEFT = (sq_X_0, sq_Y_0)
sq_cd_UP_RIGHT = (sq_X_1, sq_Y_0)
sq_cd_DN_LEFT = (sq_X_0, sq_Y_1)
sq_cd_DN_RIGHT = (sq_X_1, sq_Y_1)


def merge_base_images(null_img, merging_img, type):
    heigh_n, width_n = null_img.shape[:2]
    heigh_m, width_m, channels = merging_img.shape
    cut_n = null_img[0:sq_Y_0, sq_X_0:sq_X_1]


def image_slitching(img_null, img_N, img_S, img_W, img_E):
    cv2.line(img_null, sq_cd_UP_LEFT, sq_cd_UP_RIGHT, (255, 255, 0), 1) if DB_show_square_lines == 1 else 0
    cv2.line(img_null, sq_cd_UP_LEFT, sq_cd_DN_LEFT, (255, 255, 0), 1) if DB_show_square_lines == 1 else 0
    cv2.line(img_null, sq_cd_UP_RIGHT, sq_cd_DN_RIGHT, (255, 255, 0), 1) if DB_show_square_lines == 1 else 0
    cv2.line(img_null, sq_cd_DN_LEFT, sq_cd_DN_RIGHT, (255, 255, 0), 1) if DB_show_square_lines == 1 else 0

    # creating North part
    img_N_cut = img_N[0:sq_Y_0, sq_X_0:sq_X_1]
    cv2.imshow('img_N_cut', img_N_cut) if DB_show_img == 1 else 0
    merge_base_images(img_null, img_N_cut, "North")

    # creating South part
    img_S_cut = img_S[sq_Y_1:1080, sq_X_0:sq_X_1]
    cv2.imshow('img_S_cut', img_S_cut) if DB_show_img == 1 else 0
    # creating West part
    img_W_cut = img_W[sq_Y_0:sq_Y_1, 350:sq_X_0]  # 350 bad code
    cv2.imshow('img_W_cut', img_W_cut) if DB_show_img == 1 else 0
    # creating East part
    img_E_cut = img_E[sq_Y_0:sq_Y_1, sq_X_1:sq_X_1 + 300]  # 300 bad code
    cv2.imshow('img_E_cut', img_E_cut) if DB_show_img == 1 else 0




if __name__ == '__main__':
    img_null = cv2.imread('../../out/13/PS/null.jpg')
    img_N = cv2.imread('../../out/13/PS/N_PS.png')
    img_S = cv2.imread('../../out/13/PS/S_PS.png')
    img_W = cv2.imread('../../out/13/PS/W_PS.png')
    img_E = cv2.imread('../../out/13/PS/E_PS.png')

    image_slitching(img_null, img_N, img_S, img_W, img_E)

    # cv2.imshow('img_null', cv2.resize(img_null, (1000, 563)))

    cv2.waitKey()
