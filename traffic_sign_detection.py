###############################################################################
###############################################################################

# Lets create an rectangle which can slide on image pixel by pixel to find
# and take inside keypoints...

###############################################################################
###############################################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression

###############################################################################
###############################################################################

def kp_coordinate(image, template_img, kp1, des1, kp2, des2, threshold):
    
    #--------------------------------------------------------------------------
    h, w, d = image.shape
    mask_image = np.zeros((h,w)) # mask_image[int(key_points[i][1]),int(key_points[i][0])] = 1
    keyPoints = []
    #--------------------------------------------------------------------------
    matches = matcher.knnMatch(des1, des2, 2)
    
    good = []
      
    for m,n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])
            
            coordinates = kp1[m.queryIdx].pt
            cv2.circle(image, (int(coordinates[0]), int(coordinates[1])), 5, (255, 0, 255), -1)
    #--------------------------------------------------------------------------            
            keyPoints.append([coordinates[0],coordinates[1]])
                 
    for i in range(len(keyPoints)):
        mask_image[int(keyPoints[i][1]),int(keyPoints[i][0])] = 1
                
    return mask_image, keyPoints
                
###############################################################################
###############################################################################

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

img1 = cv2.imread("/home/artint/Downloads/FullIJCNN2013/00888.ppm")
img2 = cv2.imread("/home/artint/Pictures/Screenshot from 2020-06-29 09-14-52.png")

keyP1, desc1 = sift.detectAndCompute(img1, None)
keyP2, desc2 = sift.detectAndCompute(img2, None)

#------------------------------------------------------------------------------

mask_image, keyPoints = kp_coordinate(img1, img2, keyP1, desc1, keyP2, desc2, 0.5)

plt.figure(0)
plt.imshow(mask_image)

#------------------------------------------------------------------------------

slide_window_height, slide_window_width = 200, 200
slide_window_stepSize = 50

# slide Window movement is 
# 1- start top-left of first-row and go through right-top of first-row
# 2- and after that starts top-left of second-row and go through right-top of second-row.
# movement is continues like that untill the end of last-row's last coordinates.

results = []

for y1 in range(0, mask_image.shape[0], slide_window_stepSize): # y1 = y1 + slide_window_stepSize
    for x1 in range(0, mask_image.shape[1], slide_window_stepSize): # x1 = x1 + slide_window_stepSize
       
        # we know (x1,y1) point
        # but we should calculate (x2,y2) point each movement of slide window or (x1,y1).
        
        y2 = y1 + slide_window_height
        x2 = x1 + slide_window_width
        
        # we need (x2,y2) to count keypoint of inside slide window.
        
        points = [ [x,y] for [x,y] in keyPoints if y>y1 and y<y2 and x>x1 and x<x2]
        
        if (len(points) >= 35):
            results.append([x1, y1, x2, y2])
            
np_array_rects = np.array([[x1,y1,x2,y2] for (x1,y1,x2,y2) in results])
rects = non_max_suppression(np_array_rects, probs = None, overlapThresh = 0.1)
            
for x1, y1, x2, y2 in rects:
    cv2.rectangle(img1, (x1,y1), (x2,y2), (0,0,255),3 )
            
plt.figure(1)
plt.imshow(img1)

plt.figure(2)
plt.imshow(img2)

############################################################################### 
######### we applied sliding window on image and found keypoints ##############
############################################################################### 