import cv2
import cvzone as cv
from cv2 import threshold
from matplotlib.pyplot import contour

cap_video = cv2.VideoCapture("eye_recording.flv")

while True:
    ret, frame = cap_video.read()
    if ret is False:
        break
    frame_size = frame[269:765, 537:1416]
    rows, cols, _ = frame_size.shape
    gray_frame_size = cv2.cvtColor(frame_size, cv2.COLOR_BGR2GRAY)
    gray_frame_size = cv2.GaussianBlur(gray_frame_size, (7, 7), 0)
    _, threshold_frame_size = cv2.threshold(gray_frame_size, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_frame_size, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        # cv2.drawContours(frame_size, [cnt], -1, (0, 255, 0), 3)
        cv2.rectangle(frame_size, (x,y), (x+w, y+h), (0, 255, 0), 3)
        # cv2.line(frame_size, (x+(w/2),0), (x+(w/2), rows), (0, 0, 0), 2)
        cv2.line(frame_size,(x+int(w/2),0), (x+int(w/2),rows) ,(0,0,255),2)
        cv2.line(frame_size,(0,y+int(h/2)), (cols,y+int(h/2)) ,(0,0,255),2)
        
        cv2.rectangle(gray_frame_size, (x,y), (x+w, y+h), (0, 255, 0), 3)
        # cv2.line(frame_size, (x+(w/2),0), (x+(w/2), rows), (0, 0, 0), 2)
        cv2.line(gray_frame_size,(x+int(w/2),0), (x+int(w/2),rows) ,(0,0,255),2)
        cv2.line(gray_frame_size,(0,y+int(h/2)), (cols,y+int(h/2)) ,(0,0,255),2)
        break
    
    
    cv2.imshow("Threshold of the image", threshold_frame_size)
    cv2.imshow("Moving of the image ", gray_frame_size)
    cv2.imshow("Original image ", frame_size)
    key = cv2.waitKey(30)
    if key == 27:
        break
cv2.destroyAllWindows()
