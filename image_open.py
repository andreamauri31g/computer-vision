import cv2

img = cv2.imread('raccoon.jpg')
img2 = cv2.imread('raccoon.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imshow('img2', img2)
cv2.waitKey(0)

cv2.destroyAllWindows()
