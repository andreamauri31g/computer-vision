import cv2
import sys
from datetime import datetime

rec = False
mode_1 = False
mode_2 = False

video_capture = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'MJPG')
out = None

b, _ = video_capture.read()

if not b:
    print('Error')
    sys.exit()

while video_capture.isOpened():
    _, frame = video_capture.read()

    if mode_1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode_2:
        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        cv2.putText(frame, now, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    if rec:
        out.write(frame)
        cv2.circle(frame, (frame.shape[1] - 30, frame.shape[0] - 30), 10, (0, 0, 255), cv2.FILLED)

    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break
    elif k == ord('1'):
        mode_1 = not mode_1
    elif k == ord('2'):
        mode_2 = not mode_2
    elif k == ord('3'):
        name = datetime.now().strftime('%Y%m%d%H%M%S%f') + '.jpg'
        cv2.imwrite(name, frame)
    elif k == ord('4'):
        if out == None:
            out = cv2.VideoWriter('output.avi', codec, 20, (640, 480))
        rec = not rec

if out != None:
    out.release()
video_capture.release()
cv2.destroyAllWindows()
