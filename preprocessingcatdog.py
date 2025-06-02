import cv2
import os

DATASET_DIR = "datasets/catdog/"
DATASET_NAME = "catdog.csv"
SIZE = (64, 64)

csv = open(DATASET_DIR + DATASET_NAME, "w")

classes = {'cat': '1', 'dog': '0'}
counter = {'cat': 0, 'dog': 0}

for c in classes:
    current_dir = DATASET_DIR + c
    for img in os.listdir(current_dir):
        if ".jpg" not in img:
            continue

        im = cv2.imread(current_dir + "/" + img, cv2.IMREAD_GRAYSCALE)

        try:
            im = cv2.resize(im, SIZE)
        except Exception as e:
            print(f"Error -> {img}")
            continue

        array = im.flatten()
        s = ",".join(array.astype(str))
        csv.write(s + "," + classes[c] + "\n")

        counter[c] += 1

csv.close()

print("Done!")
print(counter)
print(counter['cat'] + counter["dog"])
