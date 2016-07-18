require 'torch'
require 'image'
require 'nn'
--[[
cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output
]]--

-- Aruns custom functions
dofile("./cv.lua")

file = 'images/lcn_12_realigned.png'
file = 'images/lcn13.png'
file = 'images/lcnstest2.png'
img1 = image.load(file)
img = img1


gray = img[1]
cropped = gray[{{600,800},{400,600}}] -- [{{220,500},{450,850}}]
img = cropped

-- itorch.image(img)
-- image.display(img)

function cv.thresh(img, f)
    local f = f or 0.5
    local ret,th1 = cv.threshold{cv.byteify(img),nil,f * 255,255,cv.THRESH_BINARY}
    return th1:double()/255
end


-- itorch.image(cv.thresh(img,0.5))
-- image.display(cv.thresh(img,0.5))


th3 = cv.adaptiveThreshold{
                src = cv.byteify(img),
                maxValue = 255,
                adaptiveMethod = cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType = cv.THRESH_BINARY,
                blockSize = 11,
                C = 6}

-- itorch.image(th3)
-- image.display(th3)

--[[
image.display(cv.Canny{
        image = cv.byteify(img), 
        threshold1 = 200, 
        threshold2 = 255})
]]--


img = img1
img = img[{1,{},{1,900}}]
-- image.display(img)

edges = cv.Canny{
        image = cv.byteify(img), 
        threshold1 = 100, 
        threshold2 = 255}
lines = cv.HoughLines{image = edges,
            rho = 1,
            theta = 3.14/180,
            threshold = 50}

-- image.display(edges)

lines  = lines[{{},1,{}}]
print(lines)
for i,p in ipairs(torch.totable(lines)) do 
    rho,theta = p[1],p[2]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = tonumber(x0 + 900*(-b))
    y1 = tonumber(y0 + 900*(a))
    x2 = tonumber(x0 - 900*(-b))
    y2 = tonumber(y0 - 900*(a))
    --print(rho,x1,y1,x2,y2)
    cv.line{
        img = img,
        pt1 = cv.Point(x1,y1),
        pt2 = cv.Point(x2,y2),
        color = cv.Scalar(0,0,255),
        thickness = 1}
    --itorch.image(img)
end

-- itorch.image(img)
-- image.display(img)


-- Hough Lines 2


-- gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img1 = image.load(file)
img = img1
gray = img[1]
img = img[{1,{},{1,900}}]
gray = img
edges = cv.Canny{ cv.byteify( gray ), 50, 150, apertureSize = 3 }
edges1 = edges
-- image.display(edges)
minLineLength = 10
maxLineGap = 3
lines = cv.HoughLinesP{ edges, 1, 3.14/180, 40, minLineLength, maxLineGap }
for i, p in ipairs(torch.totable(lines)) do
  print( p[1][1], p[1][2], p[1][3], p[1][4])
  x1, y1, x2, y2 = p[1][1], p[1][2], p[1][3], p[1][4]
  cv.line{ img, cv.Point( x1, y1 ) , cv.Point( x2, y2 ), cv.Scalar(0, 255, 0), 30 }
  cv.line{ edges1, cv.Point( x1, y1 ) , cv.Point( x2, y2 ), cv.Scalar(0, 255, 0), 30 }
end

-- cv2.imwrite('houghlines5.jpg',img)
image.display(img)
image.display(edges1)

--[[

]]--

