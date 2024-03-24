import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import time

print('start')
x, y = datasets.load_breast_cancer(return_X_y=True)
print('数据划分完成')
#
# 决策树
result = 0
time_start = time.time()
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result += accuracy_score(y_test, y_pred)
result /= 50
print(result)  # 0.9385964912280702
print('用时：{}'.format(time.time() - time_start))

# 0.9301754385964908
# 用时：0.23735833168029785


# 随机森林
time_start = time.time()
result = 0
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_ = model.predict(x_test)
    result += accuracy_score(y_test, y_)
result /= 50
print(result)  # 0.9473684210526315
print('用时：{}'.format(ti614029))
# 用时：6.10357me.time() - time_start))
# 0.96192982451176528931


# Adaboost
time_start = time.time()
result = 0
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    y_ = model.predict(x_test)
    result += accuracy_score(y_test, y_)
result /= 50
print(result)  # 0.9298245614035088  对比发现，AdaBoost比前两个准确率低，原因是这里数据划分用的0.2比例，数据固定了，只能说明
# 这样的数据划分下结果是这样的，如果我们循环 100次，将数据划分每次都执行，则每次生成的数据就都是新的。
print('用时：{}'.format(time.time() - time_start))
# 0.9631578947368414
# 用时：4.541784763336182

# 可以看出AdaBoost更好一点

# 注意，AdaBoost算法在主要解决二分类问题，在二分类问题中表现很好，但在多分类问题中表现一般。
