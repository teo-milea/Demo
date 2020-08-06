# importing the required modules 
  

import cv2 
  

import numpy as np 
  
  

# capturing from the first camera attached 
  

cap = cv2.VideoCapture(0) 
  
  

# will continue to capture until 'q' key is pressed 
 

cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('gray', cv2.WINDOW_AUTOSIZE)

while True: 
    ret, frame = cap.read() 

    # Capturing in grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    cv2.imshow('frame', frame) 
    cv2.imshow('gray', gray) 

    # Program will terminate when 'q' key is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    # cv2.destroyAllWindows()

# Releasing all the resources 
cap.release() 
cv2.destroyAllWindows() 