import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from imutils.object_detection import non_max_suppression 

###############################################################################
# we will read and make useful dataset with read() function.
###############################################################################

def read(mainFolderPath):
    
    
    main_folder_list = os.listdir(mainFolderPath)   
    folders = []
    big_images =[]
    for i in main_folder_list:
        if ".ppm" in i:
            big_images.append(i)
        elif len(i) == 2:
            folders.append(i)
    
    
    small_images = []
    for folder_name in folders:
        folder_path = mainFolderPath + "/" + folder_name    
        folder_images = os.listdir(folder_path)
        count = 0
        for image_name in folder_images:
            image_path = folder_path + "/" + image_name
            image = cv2.imread(image_path)
            small_images.append([image_path, image])
            count += 1
            if count > 10:
                break

    annotations = {}
    annt_txt_path = mainFolderPath + "/" + "gt.txt"
    with open(annt_txt_path, "r") as annt:

        for line in annt:
            filename, x1, y1, x2, y2, label = line.split(";")
            
            if filename in annotations:
                annotations[filename].append([int(x1), int(y1), int(x2), int(y2)])
            else:
                annotations[filename] = [[int(x1), int(y1), int(x2), int(y2)]]
                
    return annotations, small_images, big_images

###############################################################################
# we will determine the keypoints coordinate on mask image with kp_coordinate() function.
###############################################################################

def kp_coordinate(image, template, kp1, des1, kp2, des2, threshold = 0.5):
    
    h, w, d = image.shape
    # we creates a mask image with np.zeros() (consisting of zeros(0) --->>> black mask image)
    mask_image = np.zeros((h,w))
    # coordinate_points ---> are keypoints coordinates. 
    # we gonna turn white these coordinate_points on mask_image with
    # mask_image[int(coordinate_points[i][0]) , int(coordinate_points[i][1])] = 1 code. 
    coordinate_points = []
    matches = matcher.knnMatch(des1, des2, k=2)
    # with k=2, the 2 nearest descriptor are returned.
    for i, (m,n) in enumerate(matches):
        
        # print("i:",i,"m:",m,"n:",n) # i: 4026 m: <DMatch 0x7f0cae4d15d0> n: <DMatch 0x7f0cae4d15f0>
        if m.distance < 0.5 * n.distance:
            coord1 = kp1[m.queryIdx].pt
            # coord2 = kp2[m.trainIdx].pt
            # print("coordinate 1:", coord1, "coordinate 2:", coord2)
            # --->>> coordinate 1: (688.9403686523438, 449.42364501953125) coordinate 2: (36.263038635253906, 27.517852783203125)
            # --->>> coordinate 1: (857.5438842773438, 438.4954528808594) coordinate 2: (36.263038635253906, 27.517852783203125)
            # --->>> coordinate 1: (1159.298095703125, 709.005615234375) coordinate 2: (36.263038635253906, 27.517852783203125)
            # --->>> coordinate 1: (75.21812438964844, 670.3236694335938) coordinate 2: (42.15278625488281, 41.914546966552734)
            coordinate_points.append([coord1[0],coord1[1]])
            
    for i in range(len(coordinate_points)):
        mask_image[int(coordinate_points[i][1]) , int(coordinate_points[i][0])] = 1
        
    return mask_image, coordinate_points

# ###############################################################################
# and here main processes...
# ############################################################################### 

main_folder_path = "/home/artint/Downloads/FullIJCNN2013"
annotations, small_images, big_images = read(main_folder_path)
f = open("template_detects.txt","w+")

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})


start = timer()
kp2desc2List = []
for name, template in small_images:
    image2 = template
    kp2, desc2 = sift.detectAndCompute(image2, None)
    # if isn't there enough descriptor (desc2 is None or len(desc2)<5) skip 
    # current imaage and continue with other image...
    if desc2 is None or len(desc2)<5 :
        continue
    # otherwise we gonna add "name, template, kp2, desc2" values
    # to kp2desc2List list...
    kp2desc2List.append([name, template, kp2, desc2])
end = timer()
print("time to calculate kp2 for small images:", end-start)
print("length of small_images:",len(kp2desc2List))


for i in range(len(big_images)):   
    start = timer()
    image1 = cv2.imread(main_folder_path + "/" + big_images[i])
    kp1, desc1 = sift.detectAndCompute(image1, None)
    
    results = []
    for template_name, template, kp2, desc2 in kp2desc2List:
        mask_image, coordinate_points = kp_coordinate(image1, template.copy(), kp1, desc1, kp2, desc2, threshold = 0.5)    
        kp2Count = len(kp2)
        
        if (len(coordinate_points)) < 2 or len(coordinate_points) < (kp2Count * 0.6):
            continue
        #        cv2.waitKey(0)
        
# Sliding Window --->>> Here we gonna create an rectangle which can slide on image 
# pixel by pixel to find and take inside keypoints... 
        imgTemp = image1.copy()
        # we are taking template sizes for sliding window. 
        window_height, window_width, d = template.shape
        stepSize = 50 # int(np.min([window_height, window_width]) / 2)
        
        for y1 in range(0, mask_image.shape[0], stepSize):
            for x1 in range(0, mask_image.shape[1], stepSize):
                # we need (x2,y2) points for each new (x1,y1) points after
                #  each movement of sliding window
                x2 = x1 + window_width
                y2 = y1 + window_height
                
                # lets create "points" list which keeps (x,y) points that 
                # sliding window involves...

                points = [[x,y] for (x,y) in coordinate_points if x >= x1 and x <= x2 and y >= y1 and y <= y2]
                if len(points) >= (kp2Count * 0.6):
                    results.append([x1,y1,x2,y2])
                    # we created "results" list for keeping bounding-box coordinate
                    # which has enough keypoint coordinates inside...                  

# There may be overlap problem, we will solve this problem with non-max-suppression.
# firsly we should convert "results" list to numpy array !

    results_array = np.array([[x1,y1,x2,y2] for (x1,y1,x2,y2) in results])
    rectangle = non_max_suppression(results_array,probs = None, overlapThresh = 0.1)
    for x1,y1,x2,y2 in rectangle:
        cv2.rectangle(image1, (x1,y1), (x2,y2), (255,255,0), 3)
        f.write(big_images[i]+";"+str(x1)+";"+str(y1)+";"+str(x2)+";"+str(y2)+"\n")
    
    plt.figure(i)
    plt.imshow(image1)
    plt.show()
    end = timer()
    
    print("elapsed time:", end-start)
    print("result of non_max_suppression:", rectangle)
    
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    
    # if len(results) > 0:
    #     cv2.waitKey(0)
