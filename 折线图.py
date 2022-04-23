import matplotlib.pyplot as plt
import numpy as np
path='D:/文件/model/loss_result/loss_0317_2_2015.txt'
ind=0
a=[]
with open(path,'r') as f:
    content=f.read().split(',')
    # print(type(content)) # <class 'str'>
    # content=list(float(content))
    # print(content[0:10])
    # print(content[1205])
    for i in range(0,1205,3):
        content1=content[i][8:]
        # i+=3
        # print(content1)
        content1=float(content1)
        a.append(content1)
        # print(i)
    # print(a)
# x1=[i for i in range(100) if i%2==0]
# x1=[i for i in range(403) ]

# y1=np.random.randn(50)

x1=[i for i in range(402)]
plt.plot(x1,a,label="test",linewidth=1,color="g")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('test')
plt.legend()
plt.show()