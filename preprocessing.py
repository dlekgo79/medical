#sobel msak
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
#전체 파일 읽기
file_list_path = "C:/Users/DaHaeLee/medical/image/"
dir1_list = os.listdir(file_list_path)

# 이미지 히스토그램으로 출력

# 콘트라스트 스트레칭 적용

def contrast_stretching(img,newMin,newMax):
    ch_img = img.copy()#이미지 복
    ch_img[ch_img <= newMin] = newMin #min보다 작으면 0으로 판단
    ch_img[ch_img >= newMax] = newMax #max보다 크다면 max로 판단

    # result = (ch_img - newMin) / (newMax - newMin)*255  #
    result = (ch_img - newMin) / (newMax - newMin)#골고루 퍼트려 주기

    return result
NT_test_file ="C:/Users/DaHaeLee/medical/image/20190320_E0000189_I0001662.png"
NT_test_file ="C:/Users/DaHaeLee/medical/image/20150602_E0000009_I0000211.png"
img = cv2.imread(NT_test_file,cv2.IMREAD_COLOR)
img = contrast_stretching(img,newMin=0,newMax=200)

hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.hist(img.ravel(), 256, [0,256]); 
plt.show()
img = img.resize((256,256))

# 샤프닝 적용
def sharpening():
    NT_test_file = "C:/Users/DaHaeLee/medical/image/20151217_E0000237_I0020753.png"#이미지 파일 경로 
    # NT_test_file = "C:/Users/DaHaeLee/medical/image/20190320_E0000189_I0001662.png"
    img = cv2.imread(NT_test_file,cv2.IMREAD_COLOR)
    kernel_sh = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])#filter
    sharpened = cv2.filter2D(img,-1,kernel_sh)#(이미지,출력 영상의 데이터 타입(-1이 이미지),filtering에 사용할 mask)
    return sharpened


for a in dir1_list:
    b = file_list_path + a
    img = cv2.imread(b,cv2.IMREAD_COLOR)
    kernel_sh = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])#filter
    sharpened = cv2.filter2D(img,-1,kernel_sh)#(이미지,출력 영상의 데이터 타입(-1이 이미지),filtering에 사용할 mask)
    save_file = "C:/Users/DaHaeLee/medical/imageprocessing/"+ a
    cv2.imwrite(save_file, sharpened)


# add사용해서 밝기 조절  ------------- 효과 없음
def add():
    NT_test_file = "C:/Users/DaHaeLee/medical/image/20151217_E0000237_I0020753.png"#이미지 파일 경로 
    img = cv2.imread(NT_test_file,cv2.IMREAD_GRAYSCALE)
    dst1 = cv2.add(img,50)
    return dst1

#모폴로지 연산 이용  ------ 효과 없음
def morphology():
    NT_test_file = "C:/Users/DaHaeLee/medical/image/20160422_E0001223_I0025631.png"#이미지 파일 경로 
    img = cv2.imread(NT_test_file,cv2.IMREAD_COLOR)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    dst2 = cv2.dilate(img,k)
    return dst2

# 명암비 올리기 - 흰색과 검정색 대조 -------
def contrast():
    NT_test_file = "C:/Users/DaHaeLee/medical/image/20190320_E0000189_I0001662.png"
    img = cv2.imread(NT_test_file,cv2.IMREAD_COLOR)

    alpha1 = -0.5
    alpha2 = 1.0

    dst3 = np.clip((1+alpha1) + img - 30 + alpha1,0,255).astype(np.uint8)

    return dst3


fig = plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('전처리 전')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(sharpening())
plt.title('after')
plt.xticks([])
plt.yticks([])
plt.show()

cv2.imshow("img",sharpening())
cv2.waitKey(0)
cv2.destoryAllWindows()
