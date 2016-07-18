require 'torch'
require 'image'
require 'nn'
cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output

loadType = cv.IMREAD_UNCHANGED
img = cv.imread{'images/IMG20160623164342.jpg', loadType}
img = cv.imread{'images/IMG20160623164432.jpg', loadType}
-- image.display(img1)
local edges = torch.ByteTensor(img:size()[1], img:size()[2])
img2 = cv.Canny{image=img, edges=edges, threshold1=10, threshold2=40}

image.display(img2)
-- img2 = cv.Canny( image, torch.Tensor(1):fill(1) )
