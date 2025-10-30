import os
import cv2
for file in os.listdir("data"):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(file)