import numpy as np
import cv2
import subprocess as sp

MIN_MATCH_COUNT = 10
img1 = cv2.imread('croppedpro.png',0)          # queryImage
#img2 = cv2.imread('given.jpg',0) # trainImage

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)

# REF: http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
VIDEO_URL = "http://10.5.5.9:8080/live/amba.m3u8"

cv2.namedWindow("GoPro",cv2.WINDOW_AUTOSIZE)

pipe = sp.Popen([ FFMPEG_BIN, "-i", VIDEO_URL,
           "-loglevel", "quiet", # no text output
           "-an",   # disable audio
           "-f", "image2pipe",
           "-pix_fmt", "bgr24",
           "-vcodec", "rawvideo", "-"],
           stdin = sp.PIPE, stdout = sp.PIPE)


while True:
    raw_image = pipe.stdout.read(320*240*3) # read 432*240*3 bytes (= 1 frame)
    img2 =  np.fromstring(raw_image, dtype='uint8').reshape((240,320,3))

    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    for m,n in matches:
      if m.distance < 0.75*n.distance:
        #good.append([m])
        good.append(m)

    if len(good) > MIN_MATCH_COUNT:
      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      if mask is not None:
          matchesMask = mask.ravel().tolist()

          h,w = img1.shape
          pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
          dst = cv2.perspectiveTransform(pts,M)

          img3 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

          draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                             singlePointColor = None,
                             matchesMask = matchesMask, # draw only inliers
                             flags = 2)

          img4 = cv2.drawMatches(img1, kp1, img3, kp2, good, None, **draw_params)

          sz = img2.shape
          warp = cv2.warpPerspective(img3, M, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

          cv2.imshow("GoPro", img4)
      else:
          cv2.imshow("GoPro", img2)

    else:
      print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
      matchesMask = None
      cv2.imshow("GoPro", img2)

    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()
