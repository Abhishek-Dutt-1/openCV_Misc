import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/lcnstest2.png', 0)
#img = cv2.imread('test3.jpg', 0)

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# plt.subplot(121), plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()


#### Inverse

rows, cols = img.shape
crow, ccol = rows/2, cols/2

mask = np.ones((rows,cols,2), np.uint8)
#mask = np.zeros((rows,cols,2), np.uint8)

a = 30 
b = 900
#mask[ crow - b : crow + b, ccol - b : ccol + b] = 1
mask[ crow - a : crow + a, ccol - a : ccol + a] = 0

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()

img1 = 255 * ( img_back - np.amin(img_back) ) / ( np.amax(img_back) - np.amin(img_back))

cv2.imwrite('test2fft.png', img1 )
cv2.imwrite('test2fftdiff.png', np.abs( img1 - img ) )

img_diff = cv2.imread('test2fftdiff.png', 0)
cv2.imwrite('test2fftdiff_blur.png', np.abs( img1 - img_diff ) )


## Artistic image
org = cv2.imread('test2fft.png', 0 )
laplacian = cv2.Laplacian(org, cv2.CV_64F)
cv2.imwrite('test2laplacian.png', laplacian )
laplacian = cv2.imread('test2laplacian.png', 0 )
blur = cv2.medianBlur(org, 21)
cv2.imwrite( 'lapblured.png', laplacian - blur )

