import cv2

img = cv2.imread('raccoon.jpg')

img2 = cv2.resize(img, (500, 500))

rotation_matrix = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 90, 1)
img3 = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

cv2.imshow('img2', img2)
cv2.waitKey(0)

cv2.imshow('img3', img3)
cv2.waitKey(0)

cv2.imwrite('raccoonr.jpg', img3)

cv2.destroyAllWindows()
