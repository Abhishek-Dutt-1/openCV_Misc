import numpy as np
import cv2

'''
img1 = cv2.imread('1_org.jpg', 0)   # 0 for grey scaling
# Equalizes histogram to sum up to 255
img2 = cv2.equalizeHist(img1)
cv2.imwrite( '1_org_equalizeHist.png', img2 )

# Make image binary (data loss)
img1 = cv2.imread('1_org.jpg', 0)
(thresh, im_bw) = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite( 'org_binary.png', im_bw )

# Blur the image but preserve the edges
# Works better for color images
img1 = cv2.imread('1_org.jpg')   # 0 for grey scaling
blur = cv2.bilateralFilter(img1, 9, 75, 75)
cv2.imwrite( '1_org_bilateralFilter.png', blur )


# Gaussian filtering
img1 = cv2.imread('1_org.jpg', 0)   # 0 for grey scaling
gf1 = cv2.GaussianBlur(img1, (201, 201), 0)
cv2.imwrite( '1_org_GaussianBlur.png', gf1 )

# Gaussian filtering 2
img1 = cv2.imread('1_org.jpg', 0)   # 0 for grey scaling
gf2 = cv2.GaussianBlur(img1, (11, 11), 0)
cv2.imwrite( '1_org_GaussianBlur2.png', gf2 )
cv2.imwrite( '1_org_GaussianBlur_diff.png', gf2 - gf1 )


edges = gf1 - img1
cv2.imwrite( 'test.png', edges )
cv2.imwrite( 'no_edges.png', edges - img1 )
'''

####
'''
img1 = cv2.imread('1_org.jpg', 0)   # 0 for grey scaling
img2 = cv2.imread('2_test_realigned.png', 0)   # 0 for grey scaling
img3 = cv2.imread('test2_realigned.png', 0)   # 0 for grey scaling
gf1 = cv2.GaussianBlur(img1, (1, 1), 0)
gf2 = cv2.GaussianBlur(img2, (1, 1), 0)
gf3 = cv2.GaussianBlur(img3, (1, 1), 0)
cv2.imwrite( 'test_diff.png', gf2 - img1 )
cv2.imwrite( 'test2_diff.png', gf3 - img1 )
'''

img1 = cv2.imread('1_org.jpg', 0)   # 0 for grey scaling
img1 = cv2.equalizeHist(img1)
cv2.imwrite( 'hist.png', img1 )
img2 = cv2.imread('test2_realigned.png', 0)   # 0 for grey scaling
img3 = cv2.imread('test3_realigned.png', 0)   # 0 for grey scaling
gf1 = cv2.GaussianBlur(img1, (11, 11), 0)
gf2 = cv2.GaussianBlur(img2, (11, 11), 0)
gf21 = cv2.GaussianBlur(img2, (25, 25), 0)
gf3 = cv2.GaussianBlur(img3, (11, 11), 0)


cv2.imwrite( 'noblur.png', img2 - img1 )
cv2.imwrite( 'blur.png', img2 - gf1 )






