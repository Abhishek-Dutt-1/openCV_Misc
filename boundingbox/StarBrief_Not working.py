import numpy as np
import cv2
#import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('croppedpro.png',0)          # queryImage
img2 = cv2.imread('given.jpg',0) # trainImage

# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create()

# Initiate STAR detector
#star = cv2.FeatureDetector_create("STAR")
star = cv2.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp1 = star.detect(img1,None)
kp2 = star.detect(img2,None)

# compute the descriptors with BRIEF
kp1, des1 = brief.compute(img1, kp1)
kp2, des2 = brief.compute(img2, kp2)

#print brief.getInt('bytes')
print des1.shape
print des2.shape


# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        #good.append(m)

# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)
cv2.imwrite('yolo.png', img3)
#plt.imshow(img3),plt.show()


# --

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.imwrite('yolo1.png', img3)
# plt.imshow(img3, 'gray'),plt.show()
