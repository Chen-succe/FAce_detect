import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Patch3DCollection

X = np.linspace(0, 10, num=30).reshape(-1, 1)
print(X, X.shape)

# 斜率和截距随机生成
w = np.random.randint(1, 5, size=1)  # size=1  [2] 如果size=2, 是这样的[1 2]
b = np.random.randint(1, 10, size=1)
print(w, b, end='  ')  # [2] [5]

# 根据一元一次方程计算目标值y,并加上“噪声”
y = X*w + b + np.random.randn(30, 1)
# plt.scatter(X, y)
# plt.show()

# 重新构造x， b截距，相当于系数w0，前面统一乘1
X = np.concatenate([X, np.full(shape=(30,1), fill_value=1)], axis=1)
print(X, X.shape) # (30, 2)

# 正规方程求解
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y).round(2)
# print(theta)
print('一元一次真实值', w, b)
print('正规方程求解的斜率和截距', theta)
# plt.plot(X[:, 0], X.dot(theta), color='green', )
# plt.plot(X[:, 0], X.dot([w, b]), color='red')
# plt.show()

fig = plt.figure(figsize=(9, 6))
ax = Patch3D(fig)

x = np.linspace(-150, 150, 100)
y = np.linspace(0, 300, 100)
z = np.linspace(0, 100, 100)
ax.scatter(x, y, z)
plt.plot(x, y, z, color='red')
plt.show()