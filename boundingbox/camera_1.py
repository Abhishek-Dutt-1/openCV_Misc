import numpy
import cv2
import subprocess as sp

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
    raw_image = pipe.stdout.read(432*240*3) # 1080p read 432*240*3 bytes (= 1 frame)
    #raw_image = pipe.stdout.read(320*240*3) # 960p # read 432*240*3 bytes (= 1 frame)
    image =  numpy.fromstring(raw_image, dtype='uint8').reshape((240,432,3))
    #image =  numpy.fromstring(raw_image, dtype='uint8').reshape((240,320,3))
    cv2.imshow("GoPro",image)
    if cv2.waitKey(5) == 27:
        cv2.imwrite("pro.png", image)
        break
cv2.destroyAllWindows()
