import numpy as np
import cv2

img1 = cv2.imread('1_org.jpg', 0)   # 0 for grey scaling
img2 = cv2.imread('2_test_realigned.png', 0)

fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(img1)
cv2.imwrite('2_test_bgSubtracted.png', fgmask)

cv2.imwrite('2_matrix_subtracted.png', img2 - img1)

