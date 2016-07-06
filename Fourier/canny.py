import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test2fft.png', 0)
#img = cv2.imread('test2fftdiff_blur.png', 0)
edges = cv2.Canny(img, 50, 100)

'''
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''

cv2.imwrite('test2canny.png', edges )



laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
scharr = cv2.Scharr(img,cv2.CV_64F, 0, 1)


cv2.imwrite('test2laplacian.png', laplacian )
cv2.imwrite('test2sobel.png', sobel )
cv2.imwrite('test2soblex.png', sobelx )
cv2.imwrite('test2sobley.png', sobely )
cv2.imwrite('test2scharr.png', scharr )

'''
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
'''

img = cv2.imread('test2laplacian.png', 0)
median = cv2.medianBlur(img, 5)
cv2.imwrite('test2laplacianblur.png', median )

####
i1 = cv2.imread('test2fft.png', 0)
i2 = cv2.imread('test2laplacian.png', 0)
blur = cv2.medianBlur(i1, 21)
cv2.imwrite('zzz.png', i2 - blur)





