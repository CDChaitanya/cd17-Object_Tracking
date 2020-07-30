# Object Tracking

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

# Define range of purple color in HSV
lower_purple = np.array([130,50,90])
upper_purple = np.array([170,255,255])

# Create empty points array
points = []

# Get default camera window size
ret , frame = cam.read()
Height , Width = frame.shape[:2]
frame_count = 0

while True:
    # Capture webcam frame
    ret , frame = cam.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(src=hsv_img, lowerb=lower_purple, upperb=upper_purple)
    
    # Finding Contours
    contours ,_ = cv2.findContours(image=mask.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # Creating empty center array to store centroid center of mass
    center = int(Height/2) , int(Width/2)
    
    if len(contours) > 0:
        # Get the largest contour and its center
        c = max(contours , key=cv2.contourArea)
        (x , y) , radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        
        try:
            center = (int (M["m10"] / M["m00"]) , int(M["m01"] / M["m00"]) )
        except:
            center = int(Height/2) , int(Width/2)
        
        # Allow only contours that have a radiuslarger than 15 pixels
        if radius > 25:
            # Draw cirlce and leave the last center creating a trail
            cv2.circle(img=frame, center=(int(x),int(y)), radius=int(radius), color=(0,0,255), thickness=2 )
            cv2.circle(img=frame, center=center, radius=5, color=(0,255,0), thickness=-1 )
            
    # Log center points 
    points.append(center)
    
    #loop over the set of tracked points
    if radius > 25:
        for i in range(1 , len(points)):
            try:
                cv2.line(img=frame, pt1=points[i-1], pt2=points[i], color=(0,255,0), thickness=2)
            except:
                pass
            
        # making frame count =0
        frame_count=0
    else:
        # Count frame
        frame_count += 1
        
        # If we count 10 frames without object lets delete our trail
        if frame_count == 10:
            points=  []
            # when frame_count reaches 20 let's clear our trail 
            frame_count = 0
        
    # Display our object tracker
    frame = cv2.flip(frame , 1)
    cv2.imshow('Object Tracker' , frame)
    
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()
        