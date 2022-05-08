import cv2
import numpy as np
import matplotlib.pyplot as plt

# 0表示以灰度读出来
img=cv2.imread('1.jpg',0)


# 利用sobel方法进行边缘检测
# img表示原图像，即进行边缘检测的图像
# cv2.CV_64F 表示64位浮点数
# 这里不适用numpy.float64  因为可能会发生溢出。用cv的数据则会自动检测
# 1,0 参数分别表示对x轴和y轴方向的导数，即dx，dy，对于图像来说是差分，这里1表示对x求偏导，0表示不对y求导。其中，x还可以求2次导
# 对x求导就是检测x方向是否有边缘。
# 第五个参数ksize指核的大小。
# 前四个参数没有赋值，第五个参数有赋值，其他参数自己去查想了解的话。
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)


# 对y方向进行边缘检测
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

sobelXY=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
# 这里对两个方向同时进行检测，则会过滤掉仅仅是是x和y方向上的边缘


# 图像展示
# 使用的是subplot和title方法
plt.subplot(2,2,1)
plt.imshow(img)
# 其中，gray表示将图片用灰度形式显示，用引号是因为这个参数是string类型
#  也可以这样显示：print（type（‘gray））
plt.title('src')

plt.subplot(2,2,2)
plt.imshow(sobelx,cmap=plt.cm.gray_r)
plt.title('sobelX')

plt.subplot(2,2,3)
plt.imshow(sobely,cmap=plt.cm.gray)
plt.title('sobley')

plt.subplot(2,2,4)
plt.imshow(sobelXY,cmap='gray')
plt.title('sobelxy')

plt.show()

# 原博客是plt.imshow(sobelx,'gray') 但是貌似没有gray这个参数，我删了，就不是灰度图像展示了。
# 原来是需要加上参数cmap，以上是三种实现灰度图片展示的格式。
