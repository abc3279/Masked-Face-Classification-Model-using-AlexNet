import numpy as np
import glob
import cv2

paths = glob.glob('./227x227/with_mask/*.jpg') # glob.glob 경로 가져오기

x = [cv2.imread(paths[i]) for i in range(len(paths))] #이미지 읽기

for i in range(len(paths)):
    # 이미지를 불러옵니다. 예시 이미지 파일명을 적절히 변경하세요.
    image = x[i]

    # 이미지를 float32 타입으로 변환합니다.
    image = image.astype(np.float32) / 255.0

    # PCA Color Augmentation을 위한 랜덤 변수 생성
    alpha = np.random.normal(loc=0, scale=0.1, size=(1, 1, 3)) # loc: 평균, scale: 표준편차 혹은 분산.

    # 랜덤 변수를 원래의 이미지에 더해줍니다.
    augmented_image = image + alpha

    # 이미지 값이 0보다 작으면 0으로, 1보다 크면 1로 클리핑합니다.
    augmented_image = np.clip(augmented_image, 0, 1)

    # 다시 이미지를 0-255 범위로 스케일링하고 uint8로 변환합니다.
    augmented_image = (augmented_image * 255).astype(np.uint8)

    # 결과 이미지를 저장하거나 표시합니다.
    cv2.imwrite('./pcaFolder/with_mask/'+ 'PCA' + str(i) +'.jpg', augmented_image)

################################################################################################

paths = glob.glob('./227x227/without_mask/*.jpg') # glob.glob 경로 가져오기

x = [cv2.imread(paths[i]) for i in range(len(paths))] #이미지 읽기

for i in range(len(paths)):
    # 이미지를 불러옵니다. 예시 이미지 파일명을 적절히 변경하세요.
    image = x[i]

    # 이미지를 float32 타입으로 변환합니다.
    image = image.astype(np.float32) / 255.0

    # PCA Color Augmentation을 위한 랜덤 변수 생성
    alpha = np.random.normal(loc=0, scale=0.1, size=(1, 1, 3)) # loc: 평균, scale: 표준편차 혹은 분산.

    # 랜덤 변수를 원래의 이미지에 더해줍니다.
    augmented_image = image + alpha

    # 이미지 값이 0보다 작으면 0으로, 1보다 크면 1로 클리핑합니다.
    augmented_image = np.clip(augmented_image, 0, 1)

    # 다시 이미지를 0-255 범위로 스케일링하고 uint8로 변환합니다.
    augmented_image = (augmented_image * 255).astype(np.uint8)

    # 결과 이미지를 저장하거나 표시합니다.
    cv2.imwrite('./pcaFolder/without_mask/'+ 'PCA' + str(i) +'.jpg', augmented_image)


cv2.destroyAllWindows()

