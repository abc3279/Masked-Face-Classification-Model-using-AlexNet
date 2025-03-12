import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

paths = glob.glob('./227+crop/with_mask/*.jpg') # glob.glob 경로 가져오기

x = [cv2.imread(paths[i]) for i in range(len(paths))] #이미지 읽기

for i in range(len(paths)): 
    flippedImage = cv2.flip(x[i], 1)
    cv2.imwrite('./flipFolder/with_mask/'+ 'F' + str(i) +'.jpg', flippedImage)

print("With Mask Flip End")


paths = glob.glob('./227+crop/without_mask/*.jpg') # glob.glob 경로 가져오기

x = [cv2.imread(paths[i]) for i in range(len(paths))] #이미지 읽기

for i in range(len(paths)): 
    flippedImage = cv2.flip(x[i], 1)
    cv2.imwrite('./flipFolder/without_mask/'+ 'F' + str(i) +'.jpg', flippedImage)

print("Without Mask Flip End")

cv2.destroyAllWindows()

