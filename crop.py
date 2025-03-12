import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

paths = glob.glob('./256x256/with_mask/*.jpg') # glob.glob 경로 가져오기

x = [cv2.imread(paths[i]) for i in range(len(paths))] #이미지 읽기

# 좌측상단
for i in range(len(paths)): 

    left = 0
    bottom = 227
    right = 227
    top = 0
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/with_mask/'+ 'LT' + str(i) +'.jpg', cropImage)

print("LT End")

# 좌측하단
for i in range(len(paths)): 

    left = 0
    bottom = 256
    right = 227
    top = 29
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/with_mask/'+ 'LB' + str(i) +'.jpg', cropImage)

print("LB End")

# 우측상단
for i in range(len(paths)): 

    left = 29
    bottom = 227
    right = 256
    top = 0
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/with_mask/'+ 'RT' + str(i) +'.jpg', cropImage)

print("RT End")

# 우측하단
for i in range(len(paths)): 

    left = 29
    bottom = 256
    right = 256
    top = 29
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/with_mask/'+ 'RB' + str(i) +'.jpg', cropImage)

print("RB End")

# 중앙
for i in range(len(paths)): 

    left = 14
    bottom = 242
    right = 241
    top = 15
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/with_mask/'+ 'C' + str(i) +'.jpg', cropImage)

print("C End")


####################################################################################


paths = glob.glob('./256x256/without_mask/*.jpg') # glob.glob 경로 가져오기

x = [cv2.imread(paths[i]) for i in range(len(paths))] #이미지 읽기

# 좌측상단
for i in range(len(paths)): 

    left = 0
    bottom = 227
    right = 227
    top = 0
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/without_mask/'+ 'LT' + str(i) +'.jpg', cropImage)

# 좌측하단
for i in range(len(paths)): 

    left = 0
    bottom = 256
    right = 227
    top = 29
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/without_mask/'+ 'LB' + str(i) +'.jpg', cropImage)


# 우측상단
for i in range(len(paths)): 

    left = 29
    bottom = 227
    right = 256
    top = 0
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/without_mask/'+ 'RT' + str(i) +'.jpg', cropImage)

# 우측하단
for i in range(len(paths)): 

    left = 29
    bottom = 256
    right = 256
    top = 29
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/without_mask/'+ 'RB' + str(i) +'.jpg', cropImage)

# 중앙
for i in range(len(paths)): 

    left = 14
    bottom = 242
    right = 241
    top = 15
    
    cropImage = (x[i])[top:bottom, left:right]
    cv2.imwrite('./cropFolder/without_mask/'+ 'C' + str(i) +'.jpg', cropImage)

cv2.destroyAllWindows()
