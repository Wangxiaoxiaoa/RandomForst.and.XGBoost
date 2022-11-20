import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from XGB_model import XGB
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
x = pd.DataFrame(cancer.data)
print(x.columns)
y = pd.Series(cancer.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
print(x_train.shape,y_train.shape,type(x_train),type(x_test))

#设置分值记录
train_f1 = []
test_f1 = []
lr = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in range(len(lr)):
    print('learning_rate为{}:'.format(lr[i]))
    xgb = XGB(base_score=0.5,max_depth=6,n_estimators=4,learning_rate=lr[i],reg_lambda=1,
              gamma=0,min_child_sample=1,min_child_weight=1,objective='classifier')
    xgb.fit(x_train, y_train)
    # 训练集评估
    train_predict = xgb.predict_prob(x_train)
    train_f1_score = f1_score(y_train, train_predict)
    train_f1.append(train_f1_score)
    print('learning_rate为{}时，训练集f1_score为{}：'.format(lr[i], train_f1_score))
    # 测试集评估
    test_predict = xgb.predict_prob(x_test)
    test_f1_score = f1_score(y_test, test_predict)
    test_f1.append(test_f1_score)
    print('learning_rate为{}时，测试集f1_score为{}：'.format(lr[i], test_f1_score))

# 图形展示
plt.plot(lr,train_f1,'b', label='train_f1_score')
plt.plot(lr,test_f1,'r', label='test_f1_score')
plt.legend(loc='best')
# plt.xlim(0,1)
plt.xticks(lr)
plt.xlabel('learning_rate')
plt.ylabel('f1_score')
plt.title('learning_rate and f1_score')
plt.show()