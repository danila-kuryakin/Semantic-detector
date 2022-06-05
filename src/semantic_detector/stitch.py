import cv2
import numpy as np

drawing = False  # true if mouse is pressed
src_x, src_y = -1, -1
dst_x, dst_y = -1, -1

src_list = []
dst_list = []


# Выделение точек на изображении src
def select_points_src(event, x, y, flags, param):
  global drawing, src_x, src_y
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    src_x, src_y = x, y
    cv2.circle(s_copy, (x, y), 5, (0, 0, 255), -1)
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False


# Выделение точек на изображении dst
def select_points_dst(event, x, y, flags, param):
  global drawing, dst_x, dst_y
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    dst_x, dst_y = x, y
    cv2.circle(d_copy, (x, y), 5, (0, 0, 255), -1)
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False


# Перспективное преобразование
def get_plan_view(src, dst):
  print('Create plan view')
  src_pts = np.array(src_list).reshape(-1, 1, 2)
  dst_pts = np.array(dst_list).reshape(-1, 1, 2)
  H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  print("H:")
  print(H)
  plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
  cv2.imshow("plan view", plan_view)
  return plan_view


# Обработка получившихся швов
def clear_image(img):
  img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)
  mask_inv = cv2.bitwise_not(mask)
  img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
  return img1_bg


# Инициализация двух начальных изображений
def image_initialization():
  print('Images initialization')
  src_image = cv2.resize(cv2.imread('E_PS.jpg', -1), (800, 600))
  src_copy = src_image.copy()
  cv2.namedWindow('src')
  cv2.moveWindow("src", 80, 80)
  cv2.setMouseCallback('src', select_points_src)

  dst_image = cv2.resize(cv2.imread('W_PS.jpg', -1), (800, 600))
  dst_copy = dst_image.copy()
  cv2.namedWindow('dst')
  cv2.moveWindow("dst", 780, 80)
  cv2.setMouseCallback('dst', select_points_dst)

  return src_copy, dst_copy, src_image, dst_image


# Сохранение точек выставленных вручную
def save_points_by_hand(copy_s, copy_d):
  print('Save points')
  cv2.circle(copy_s, (src_x, src_y), 5, (0, 255, 0), -1)
  cv2.circle(copy_d, (dst_x, dst_y), 5, (0, 255, 0), -1)
  src_list.append([src_x, src_y])
  dst_list.append([dst_x, dst_y])
  print("src points:")
  print(src_list)
  print("dst points:")
  print(dst_list)


# Статическое сохранение всех необходимых точек
def save_points_static():
  print('Save points')
  src_list.append([[611, 192], [607, 234], [610, 278], [608, 319], [360, 468], [271, 300], [399, 129], [433, 468],
                   [433, 468], [277, 411], [228, 313], [220, 419], [532, 396]])

  dst_list.append([[603, 195], [606, 235], [612, 279], [613, 319], [363, 467], [272, 298], [398, 129], [398, 129],
                   [436, 468], [272, 411], [232, 310], [212, 419], [528, 413]])

  print("src points:")
  print(src_list)
  print("dst points:")
  print(dst_list)


# Сшивание и "чистка" образовавшихся швов
def merge_and_clearing(src, dst, name_file):
  print('Merge and clearing views')
  # Обработка входных изображений
  dst = clear_image(dst)
  src = clear_image(src)
  # Процесс сшивания картинок
  merge = get_plan_view(src, dst)
  for i in range(0, dst.shape[0]):
    for j in range(0, dst.shape[1]):
      if merge.item(i, j, 0) == 0 and merge.item(i, j, 1) == 0 and merge.item(i, j, 2) == 0:
        merge.itemset((i, j, 0), dst.item(i, j, 0))
        merge.itemset((i, j, 1), dst.item(i, j, 1))
        merge.itemset((i, j, 2), dst.item(i, j, 2))
  # "Чистка" швов на полученном изображении
  img2gray = cv2.cvtColor(merge, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY_INV)
  mask_inv = cv2.bitwise_not(mask)
  img1_bg = cv2.bitwise_and(merge, merge, mask=mask_inv)
  # Изменение цвета шва под "цвет дороги"
  blank_image = np.zeros((600, 800, 3), np.uint8)
  blank_image[:, 0:800] = (87, 88, 91)
  blank_image = cv2.bitwise_and(blank_image, blank_image, mask=mask)
  finally_merge = cv2.add(img1_bg, blank_image)
  cv2.imshow('final', finally_merge)
  file_path = "../../out/" + name_file + ".jpg"
  print(file_path)
  cv2.imwrite(format(file_path), finally_merge)


# Метод отслеживания нажатий на картинку и отрисовка на них точек
def images_callback():
  cv2.setMouseCallback('src', select_points_src)
  cv2.setMouseCallback('dst', select_points_dst)


name_of_stitching_file = input("Enter final file name: ")
s_copy, d_copy, src_img, dst_img = image_initialization()
"""
Клавиша 's' - сохранение нанесённых точек
Клавиша 'h' - построение перспективного преобразования
Клавиша 'm' - сшивание двух изображений (результат сохраняется в папку out) 
Клавиша 'e' - выход из программы
"""
while 1:
  cv2.imshow('src', s_copy)
  cv2.imshow('dst', d_copy)
  k = cv2.waitKey(1) & 0xFF

  images_callback()
  if k == ord('s'):
    #save_points_by_hand(s_copy, d_copy)
    save_points_static()
  elif k == ord('h'):
    mPlan_view = get_plan_view(src_img, dst_img)
  elif k == ord('m'):
    merge_and_clearing(src_img, dst_img, name_of_stitching_file)
  elif k == ord('e'):
    break

cv2.destroyAllWindows()
