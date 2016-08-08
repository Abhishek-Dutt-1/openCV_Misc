import cv2
import numpy as np

img = cv2.imread('images/test1.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print M

# Centroid
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])
#img2 = cv2.circle(img, cx, cy, 10, 0, 2)

# Area
area = cv2.contourArea(cnt)

# Contour Perimeter
perimeter = cv2.arcLength(cnt,True)

# Contour Approximation
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


