import sys
import numpy
import os
import matplotlib.pyplot as plt
from PIL import Image

img=Image.open('1.jpg')

r,g,b=img.split()
gray=r.convert('L') # 背景还是黄色

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(gray,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(g,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(b,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(img)
plt.show()

# https://blog.csdn.net/majinlei121/article/details/78935083  附上博客
