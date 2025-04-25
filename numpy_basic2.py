import numpy as np
from PIL import Image

arr = np.random.randint(0, 255, (1000, 1000))
arr_white = np.ones((500, 500)) * 255

x_offset = arr_white.shape[0] // 2
y_offset = arr_white.shape[1] // 2
x_start = arr.shape[0] // 2 - x_offset
x_end = arr.shape[0] // 2 + x_offset
y_start = arr.shape[1] // 2 - y_offset
y_end = arr.shape[1] // 2 + y_offset
arr[x_start:x_end,y_start:y_end] = arr_white

img = Image.fromarray(arr)
print(arr)
img.show()
