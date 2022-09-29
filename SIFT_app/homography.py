#!/usr/bin/env python3
import cv2
import numpy as np
import time
img = cv2.imread("000_image.jpg", cv2.IMREAD_GRAYSCALE) # queryimage
cap = cv2.VideoCapture("IMG-9532.MOV") # input 0 for webcam


# Features
sift = cv2.SIFT_create() # sift algo
kp_image, desc_image = sift.detectAndCompute(img, None) # detect keypoints, descriptors, None mask
img = cv2.drawKeypoints(img, kp_image, img)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params) # Flann algo

while True:
    ret, frame = cap.read()
    dim = (600, 600)
    frame = cv2.resize(np.flip(frame), dim, interpolation = cv2.INTER_AREA)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # trainimage
    
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good_points = []

    for m, n in matches: # m = query image, n = train image
        # ratio of how good the match is 
        if m.distance < 0.6*n.distance: # ratio test
            good_points.append(m)

    img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

    # Homography
    if (len(good_points)) > 5: # homography if at least 10 matches
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1,1,2) # extracting pos of good pts of query image
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w, c = img.shape
        pts = np.float32([[0,0], [0, h], [w,h], [w,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)

        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv2.imshow("Homography", homography)
        # cv2.imshow("img3")
    else:
        cv2.imshow("img3", img3)
        pass
        # cv2.imshow("Homography", grayframe)


    # cv2.imshow("Image", img)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("grayFrame", grayframe)
    key = cv2.waitKey(2)
    
    # 's' key on keyboard
    if key == 27:
        break


cap.release()
cap.destroyAllWindows()