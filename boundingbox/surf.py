import cv2
image = cv2.imread('images/test11.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# kps: 274, descriptors: (274, 128)

surf = cv2.xfeatures2d.SURF_create()

(kps, descs) = surf.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# kps: 393, descriptors: (393, 64)

surf.setHessianThreshold(2000)
print surf.getHessianThreshold()

(kp, des) = surf.detectAndCompute( gray, None )
print("# kps: {}, descriptors: {}".format(len(kp), des.shape))


img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)

cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check upright flag, if it False, set it to True
print surf.getUpright()
surf.setUpright(True)
print surf.getUpright()
# Recompute the feature points and draw it
kp = surf.detect( gray, None )
img2 = cv2.drawKeypoints( image, kp, None, (255,0,0), 4 )

cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find size of descriptor
print surf.descriptorSize()
# That means flag, "extended" is False.
surf.getExtended()
# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(gray,None)
print surf.descriptorSize()
print des.shape

