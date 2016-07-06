import cv2
import numpy as np

img = cv2.imread( 'lapblured.png', 0 )
kernel = np.ones( ( 3, 3 ), np.uint8 )
erode = cv2.erode( img, kernel, iterations = 1 )
dilate = cv2.dilate( img, kernel, iterations = 1 )
open = cv2.morphologyEx( img, cv2.MORPH_OPEN, kernel )
close = cv2.morphologyEx( img, cv2.MORPH_CLOSE, kernel )
grad = cv2.morphologyEx( img, cv2.MORPH_GRADIENT, kernel )
top = cv2.morphologyEx( img, cv2.MORPH_TOPHAT, kernel )
black = cv2.morphologyEx( img, cv2.MORPH_BLACKHAT, kernel )


cv2.imwrite('erode.png', erode)
cv2.imwrite('dilate.png', dilate)
cv2.imwrite('open.png', open)
cv2.imwrite('close.png', close)
cv2.imwrite('grad.png', grad)
cv2.imwrite('top.png', top)
cv2.imwrite('black.png', black)

#cv2.imshow("Image1", erosion)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
