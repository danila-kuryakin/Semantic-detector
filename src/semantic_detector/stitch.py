import cv2
import imutils
import numpy as np

drawing = False  # true if mouse is pressed
src_x, src_y = -1, -1
dst_x, dst_y = -1, -1

src_list = []
dst_list = []


# mouse callback function
def select_points_src(event, x, y, flags, param):
  global src_x, src_y, drawing
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    src_x, src_y = x, y
    cv2.circle(src_copy, (x, y), 5, (0, 0, 255), -1)
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False


# mouse callback function
def select_points_dst(event, x, y, flags, param):
  global dst_x, dst_y, drawing
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    dst_x, dst_y = x, y
    cv2.circle(dst_copy, (x, y), 5, (0, 0, 255), -1)
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False


def get_plan_view(src, dst):
  src_pts = np.array(src_list).reshape(-1, 1, 2)
  dst_pts = np.array(dst_list).reshape(-1, 1, 2)
  H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  print("H:")
  print(H)
  plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
  return plan_view


def merge_views(src, dst):
  plan_view = get_plan_view(src, dst)
  for i in range(0, dst.shape[0]):
    for j in range(0, dst.shape[1]):
      if plan_view.item(i, j, 0) == 0 and plan_view.item(i, j, 1) == 0 and plan_view.item(i, j, 2) == 0:
        plan_view.itemset((i, j, 0), dst.item(i, j, 0))
        plan_view.itemset((i, j, 1), dst.item(i, j, 1))
        plan_view.itemset((i, j, 2), dst.item(i, j, 2))
  cv2.imwrite('../../out/stitch.jpg', plan_view)
  return plan_view


def clear_image(stitch):
  img2gray = cv2.cvtColor(stitch, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)
  mask_inv = cv2.bitwise_not(mask)
  img1_bg = cv2.bitwise_and(stitch, stitch, mask=mask_inv)
  return img1_bg


src = cv2.resize(cv2.imread('S_PS.jpg', -1), (800, 600))
src_copy = src.copy()

cv2.namedWindow('src')
cv2.moveWindow("src", 80, 80)
cv2.setMouseCallback('src', select_points_src)

dst = cv2.resize(cv2.imread('../../out/clear_merge2.jpg', -1), (800, 600))
dst_copy = dst.copy()

cv2.namedWindow('dst')
cv2.moveWindow("dst", 780, 80)
cv2.setMouseCallback('dst', select_points_dst)

if __name__ == '__main__':
  while 1:
    cv2.imshow('src', src_copy)
    cv2.imshow('dst', dst_copy)
    k = cv2.waitKey(1) & 0xFF
    cv2.setMouseCallback('src', select_points_src)
    cv2.setMouseCallback('dst', select_points_dst)
    if k == ord('s'):
      print('save points')
      cv2.circle(src_copy, (src_x, src_y), 5, (0, 255, 0), -1)
      cv2.circle(dst_copy, (dst_x, dst_y), 5, (0, 255, 0), -1)
      # src_list.append([[611, 192], [607, 234], [610, 278], [608, 319], [360, 468], [271, 300], [399, 129], [433, 468], [433, 468], [277, 411], [228, 313], [220, 419], [532, 396]])
      # dst_list.append([[603, 195], [606, 235], [612, 279], [613, 319], [363, 467], [272, 298], [398, 129], [398, 129], [436, 468], [272, 411], [232, 310], [212, 419], [528, 413]])
      src_list.append([src_x, src_y])
      dst_list.append([dst_x, dst_y])
      print("src points:")
      print(src_list)
      print("dst points:")
      print(dst_list)
    elif k == ord('h'):
      print('create plan view')
      plan_view = get_plan_view(src, dst)
      cv2.imshow("plan view", plan_view)
    elif k == ord('m'):
      print('merge views')
      merge = merge_views(src, dst)
      cv2.imshow("dirty", merge)
      cv2.imwrite('../../out/merge.jpg', merge)

      dst = clear_image(dst)
      src = clear_image(src)
      merge2 = merge_views(src, dst)
      cv2.imwrite('../../out/clear_merge2.jpg', merge2)
      cv2.imshow("clear", merge2)

    elif k == ord('e'):
      break

cv2.destroyAllWindows()
