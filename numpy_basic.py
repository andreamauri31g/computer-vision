import numpy as np
from PIL import Image

# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(type(arr))
# print(arr)
# print(arr.shape)

arr = np.random.randint(0, 255, (100, 100))
img = Image.fromarray(arr)
arr[0] = np.zeros((100))
print(arr)
img.show()

# print(arr)
# print(arr[0])
# print(arr[0:5])
# print(arr[-1])
# print(arr[:,0])
# print(arr[:,0:5])
# print(arr[:,-1])
# print(arr[0,0])
