import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

print('start')
x, y = datasets.load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('数据划分完成')
#
# # 决策树
# model = DecisionTreeClassifier()
# print('开始预测')
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
#
# print('预测完成')
# result = accuracy_score(y_test, y_pred)
# print(result)  # 0.9385964912280702

# # 随机森林
# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# y_ = model.predict(x_test)
# result = accuracy_score(y_test, y_)
# print(result)  # 0.9473684210526315

# Adaboost
model = AdaBoostClassifier()
model.fit(x_train, y_train)
y_ = model.predict(x_test)
result = accuracy_score(y_test, y_)
print(result)  # 0.9298245614035088  对比发现，AdaBoost比前两个准确率低，原因是这里数据划分用的0.2比例，数据固定了，只能说明
# 这样的数据划分下结果是这样的，如果我们循环 100次，将数据划分每次都执行，则每次生成的数据就都是新的。
