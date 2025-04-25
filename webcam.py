import cv2
import sys

video_capture = cv2.VideoCapture(0)

b, frame = video_capture.read()

if not b:
    print('Error')
    sys.exit()

cv2.imshow('Frame', frame)
cv2.waitKey(0)
