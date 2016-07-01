import cv2

image = cv2.imread("test2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale

image1, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # threshold
cv2.imwrite("thresh.jpg", thresh) 
'''
image, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_TOZERO) # threshold
cv2.imwrite("contoured0.jpg", thresh) 
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) # threshold
cv2.imwrite("contoured1.jpg", thresh) 
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # threshold
cv2.imwrite("contoured2.jpg", thresh) 
'''

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3) )
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilated = cv2.dilate(thresh, kernel, iterations = 13) # dilate
dilated = cv2.dilate(thresh, kernel, iterations = 1) # dilate
cv2.imwrite("dilate.jpg", dilated) 

image1, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get contours
contoured = cv2.drawContours(gray, contours, -1, 0)
cv2.imwrite("contoured1.jpg", contoured) 

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h > 300 and w > 300:
        continue

    # discard areas that are too small
    if h < 40 or w < 40:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

# write original image with added contours to disk  
cv2.imwrite("contoured.jpg", image) 
