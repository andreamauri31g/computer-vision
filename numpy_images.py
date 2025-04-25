import numpy as np
from PIL import Image

img = Image.open('../raccoon.jpg')
img_bw = img.convert('L')
arr = np.array(img)
arr_bw = np.array(img_bw)

print(arr.shape)
print(arr_bw.shape)
print(arr_bw)
print(arr[:,:,0])
print(arr)
img.show()
img_bw.show()
