require 'torch'
require 'image'
require 'nn'

dofile("./cv.lua")

file = 'images/lcnstest2.png'
file = 'images/test2fftdiff.png'
img = image.load(file)
img = img[{1,{},{1,1200}}]
gray = img


edges = cv.Canny{ cv.byteify( gray ), 100, 150, apertureSize = 3 }
image.display(edges)
-- Detect lines
minLineLength = 20
maxLineGap = 1
lines = cv.HoughLinesP{ edges, 1, 3.14/180, 40, minLineLength, maxLineGap }
-- Detect Circles
-- circles = cv.HoughCircles{edges, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0}
-- edges, method, dp, minDistBetweenCentersOfCircle, param1=SomethingCanny, param2=Threshold, minRadiusOfCircle, maxRadiusOfCircle
circles = cv.HoughCircles{edges, cv.HOUGH_GRADIENT, 1, 1, param1=1, param2=80, minRadius=0, maxRadius=200}
circles = circles[{1}]
--[[
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
]]



file = 'images/lcnstest2.png'
img = image.load(file)
img = img[{1,{},{1,1200}}]
gray = img
edges = cv.Canny{ cv.byteify( gray ), 50, 150, apertureSize = 3 }

-- Draw lines
for i, p in ipairs(torch.totable(lines)) do
  -- print( p[1][1], p[1][2], p[1][3], p[1][4])
  x1, y1, x2, y2 = p[1][1], p[1][2], p[1][3], p[1][4]
  cv.line{ img, cv.Point( x1, y1 ) , cv.Point( x2, y2 ), cv.Scalar(0, 255, 0), 20 }
  cv.line{ edges, cv.Point( x1, y1 ) , cv.Point( x2, y2 ), cv.Scalar(0, 255, 0), 20 }
end

-- Draw Circles
for i, p in ipairs(torch.totable(circles)) do
  -- print (i, p, p[1], p[2], p[3])
  x1, y1, rad = p[1], p[2], p[3]
  print(x1, y1, rad)
  cv.circle{ img, cv.Point( x1, y1 ) , rad, cv.Scalar(0, 255, 0), 20 }
  cv.circle{ edges, cv.Point( x1, y1 ) , rad, cv.Scalar(0, 255, 0), 20 }
end

image.display(img)
image.display(edges)
