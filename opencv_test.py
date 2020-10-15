# importing the required modules 
  

import cv2 
  

import numpy as np 
  
  

# capturing from the first camera attached 
  

cap = cv2.VideoCapture(0) 
  
  

# will continue to capture until 'q' key is pressed 
 

cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

cap.set(3, 1280)
cap.set(4, 720)

while True: 
    ret, frame = cap.read() 

    frame = cv2.line(frame, (int(1280/4), 0), (int(1280/4), 720), 50, 2)

    frame = cv2.line(frame, (int(3 * 1280/4), 0), (int(3 * 1280/4), 720),50, 2)

    frame = cv2.line(frame, (0, int(720/4)), (1280, int(720/4)), 50, 2)

    frame = cv2.line(frame, (0, int(3 * 720/4)), (1280, int(3 * 720/4)), 50, 2)

    cv2.imshow('frame', frame) 
    # cv2.imshow('gray', gray) 

    # Program will terminate when 'q' key is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    # cv2.destroyAllWindows()

# Releasing all the resources 
cap.release() 
cv2.destroyAllWindows() 