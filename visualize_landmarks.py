import os
import cv2
import numpy as np
from pathlib import Path 
from PIL import Image

with open('./cosFace/data/josh_landmarksMTCNN.txt') as file:
    lines = file.readlines()

for line in lines:
    parts = line[:-1].split('\t')
    src_pts = np.array(parts[2:-1]).reshape(5,2)
    break
myface = cv2.imread(str(Path('/Users/joshuakang/git/face-detection-knn/faces/faces2/') / parts[0].split('/')[1]))
# Image.open(Path('/Users/joshuakang/git/face-detection-knn/faces/faces2/') / parts[0].split('/')[1])
cv2.imshow('lol', myface)
cv2.waitKey(0)
cv2.destroyAllWindows()