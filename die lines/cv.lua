cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output

function cv.BGR2RGB(img)
    local im = torch.zeros(img:size(3),img:size(1),img:size(2))
    im[1] = img[{{},{},3}]
    im[2] = img[{{},{},2}]
    im[3] = img[{{},{},1}]
    return im
end
function cv.RGB2BGR(img)
    local im = torch.zeros(img:size(2),img:size(3),img:size(1))
    im[{{},{},1}] = img[3]
    im[{{},{},2}] = img[2]
    im[{{},{},3}] = img[1]
    return im
end
if(itorch) then
function itorch.imagecv(img)
    itorch.image(cv.BGR2RGB(img))
end
end

function cv.bool(img)
    local tmp = img:clone()
    tmp[tmp:ne(0)] = 1
    return tmp:byte()
end

function cv.byteify(img)
    local tmp = img:clone()
    local min,max = tmp:min(),tmp:max()
    tmp = (tmp-min)/(max-min)
    tmp = 255*tmp
    return tmp:byte()
end

function cv.conn(sure_fg)
    return cv.connectedComponents{image = cv.bool(sure_fg)}
end

function cv.conn_min(img,min)
    local markers = img:clone()
    local cnts = cv.counts(markers)
    local valids = 0
    for k,v in pairs(cnts) do
        if(v>min) then 
            valids = valids + 1
        else
            markers[markers:eq(k)] = 0
        end
    end
    return markers,valids
end

function cv.subImg(markers,i)
    local tmp = markers:eq(i):nonzero()
    if(tmp:dim()==0) then return nil end
    local tmp1 = tmp:min(1)
    local tmp2 = tmp:max(1)
    return (markers[{{tmp1[1][1],tmp2[1][1]},{tmp1[1][2],tmp2[1][2]}}])
end

function cv.counts(X)
    local x = X:clone():resize(X:nElement())
    local ret = {}
    for i=1,x:size(1) do
        ret[x[i]] = ret[x[i]] and ret[x[i]] + 1 or 1
    end
    return ret
end

function cv.wshed(img,gray)
    local ret, thresh = cv.threshold{gray,nil,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU}
    local kernel = torch.ones(10,10)
    local opening = cv.morphologyEx{thresh,nil,cv.MORPH_OPEN,kernel,iterations = 2}
    local sure_bg = cv.dilate{opening,nil,kernel,iterations=3}
    local dist_transform = cv.distanceTransform{opening,nil,cv.DIST_L2,5}
    local ret, sure_fg = cv.threshold{dist_transform, nil,0.001*dist_transform:max(),255,0}
    sure_fg = sure_fg:byte()
    local unknown = sure_bg-sure_fg
    local ret,markers = cv.connectedComponents{image = 1-bool(sure_fg)}--, labels = markers}
    markers = markers+1
    markers[unknown:eq(255)] = 0
    cv.watershed{img,markers}
    return markers
end
