
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from XGB_model import XGB
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
boston = datasets.load_boston()
x = boston.data
y = boston.target
x = pd.DataFrame(boston.data)
print(x.columns)
y = pd.Series(boston.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape,y_train.shape,type(x_train),type(x_test))
#设置分值记录
train_r2 = []
test_r2 = []

for i in range(10):
    print('max_depth为{}:'.format(i+1))
    xgb = XGB(base_score=0.5,max_depth=i+1,n_estimators=12,learning_rate=0.3,reg_lambda=1,
              gamma=0,min_child_sample=1,min_child_weight=1,objective='regression')
    xgb.fit(x_train, y_train)
    #训练集评估
    train_predict = xgb.predict_raw(x_train)
    train_r2_score = r2_score(y_train,train_predict)
    train_r2.append(train_r2_score)
    print('max_depth为{}时，训练集r2_score为{}：'.format(i+1,train_r2_score))
    #测试集评估
    test_predict = xgb.predict_raw(x_test)
    test_r2_score = r2_score(y_test, test_predict)
    test_r2.append(test_r2_score)
    print('max_depth为{}时，测试集r2_score为{}：'.format(i+1,test_r2_score))

#图形展示
plt.plot(train_r2,'b',label = 'train_r2_score')
plt.plot(test_r2,'r',label = 'test_r2_score')
plt.legend(loc = 'best')
plt.xlabel('max_depth')
plt.ylabel('r2_score')
plt.title('max_depth and r2_score')
plt.show()