import cv2
import os

DATASET_DIR = "datasets/01/"
DATASET_NAME = "01.csv"

csv = open(DATASET_DIR + DATASET_NAME, "w")

counter = [0, 0]

for i in range(2):
    current_dir = DATASET_DIR + str(i)
    for img in os.listdir(current_dir):
        if ".jpg" not in img:
            continue

        im = cv2.imread(current_dir + "/" + img, cv2.IMREAD_GRAYSCALE)
        array = im.flatten()
        s = ",".join(array.astype(str))
        csv.write(s + "," + str(i) + "\n")

        counter[i] += 1

csv.close()

print("Done!")
print(counter)
