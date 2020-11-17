# 本文件完成简单预测，使用7到22点的数据(半小时步长)训练，
# 使用了线性回归模型，决策树模型，前者成绩0.15，后者0.17过拟合。
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import linear_model 
from sklearn.tree import DecisionTreeRegressor

train_data=pd.read_csv("./train_TTI.csv")
train_data['time'] = pd.to_datetime(train_data['time'])
train_data['hour'] = [int((x.hour+1)/2) for x in train_data['time']]
train_data['date'] = [datetime.strftime(x,'%Y-%m-%d') for x in train_data['time']]
grouped=train_data.groupby(['id_road','date','hour'])
#print(1,train_data)

#X=pd.DataFrame(columns=('id_road','TTI0','TTI1','TTI2','TTI3','TTI4','TTI5','speed0','speed1''speed2''speed3''speed4''speed5'))
#y=pd.DataFrame(columns=('TTI0','TTI1',"TTI2"))
X_list=[]
y0_list=[]
y1_list=[]
y2_list=[]

group_count=0
count=0
for name,group in grouped:
    # if(i==0):
    #     print(name)   # name=[275911, '2019-01-01', 0]
    #     print(name[0],name[1],name[2])
    #     print(group)
    group_count+=1
    numEachGroup=np.size(group,0)
    #print(numEachDay)
    if(name[2]<4 or name[2]>10):    # 不在7点到22点的不考虑；
        continue
    if(numEachGroup!=12):   # 有缺失值的不考虑；
        count+=1
        continue
    x_list=[]
    x_list.append(name[0])
    x_list.append(name[2])
    tti=group['TTI'].tolist()[0:6]
    y0=group['TTI'].tolist()[6]
    y1=group['TTI'].tolist()[7]
    y2=group['TTI'].tolist()[8]
    speed=group['speed'].tolist()[0:6]
    x_list.extend(tti)
    x_list.extend(speed)

    X_list.append(x_list)    # 列分别是：id_road, hour, speed0~5, TTI0~5
    y0_list.append(y0)
    y1_list.append(y1)
    y2_list.append(y2)

    x_list=[]
    x_list.append(name[0])
    x_list.append(name[2])
    tti=group['TTI'].tolist()[3:9]
    y0=group['TTI'].tolist()[9]
    y1=group['TTI'].tolist()[10]
    y2=group['TTI'].tolist()[11]
    speed=group['speed'].tolist()[3:9]
    x_list.extend(tti)
    x_list.extend(speed)

    X_list.append(x_list)    # 列分别是：id_road, hour, speed0~5, TTI0~5
    y0_list.append(y0)
    y1_list.append(y1)
    y2_list.append(y2)
print(group_count,count)


train_data['hour'] = [int((x.hour)/2) for x in train_data['time']]
#print(1,train_data)
grouped1=train_data.groupby(['id_road','date','hour'])
group_count=0
count=0
for name,group in grouped:
    group_count=group_count+1
    numEachGroup=np.size(group,0)
    #print(numEachDay)
    if(name[2]<4 or name[2]>10):    # 不在7点到22点的不考虑；
        continue
    if(numEachGroup!=12):   # 有缺失值的不考虑；
        count+=1
        continue
    x_list=[]
    x_list.append(name[0])
    x_list.append(name[2])
    tti=group['TTI'].tolist()[0:6]
    y0=group['TTI'].tolist()[6]
    y1=group['TTI'].tolist()[7]
    y2=group['TTI'].tolist()[8]
    speed=group['speed'].tolist()[0:6]
    x_list.extend(tti)
    x_list.extend(speed)

    X_list.append(x_list)    # 列分别是：id_road, hour, speed0~5, TTI0~5
    y0_list.append(y0)
    y1_list.append(y1)
    y2_list.append(y2)

    x_list=[]
    x_list.append(name[0])
    x_list.append(name[2])
    tti=group['TTI'].tolist()[3:9]
    y0=group['TTI'].tolist()[9]
    y1=group['TTI'].tolist()[10]
    y2=group['TTI'].tolist()[11]
    speed=group['speed'].tolist()[3:9]
    x_list.extend(tti)
    x_list.extend(speed)

    X_list.append(x_list)    # 列分别是：id_road, hour, speed0~5, TTI0~5
    y0_list.append(y0)
    y1_list.append(y1)
    y2_list.append(y2)
print(group_count,count)

X=np.array(X_list)
y0=np.array(y0_list)
y1=np.array(y1_list)
y2=np.array(y2_list)
# 构造X,y完毕；

# 使用线性模型；
model0 = linear_model.LinearRegression()
model1 = linear_model.LinearRegression()
model2 = linear_model.LinearRegression()
# model0.fit(X, y0)
# model1.fit(X, y1)
# model2.fit(X, y2)
#model0=DecisionTreeRegressor(criterion='mae',splitter='random',max_depth=10,min_samples_split=5,min_samples_leaf=4)
#model1=DecisionTreeRegressor(criterion='mae',splitter='random',max_depth=10,min_samples_split=5,min_samples_leaf=4)
#model2=DecisionTreeRegressor(criterion='mae',splitter='random',max_depth=10,min_samples_split=5,min_samples_leaf=4)
model0.fit(X,y0)
print(1)
model1.fit(X,y1)
print(2)
model2.fit(X,y2)
print(3)

y0_predict=model0.predict(X)
y1_predict=model1.predict(X)
y2_predict=model2.predict(X)

err0=np.abs(y0_predict-y0)
print('y0总数 = ',np.size(y0),'误差0 =',np.sum(err0),'平均误差',np.sum(err0)/np.size(y0))
err1=np.abs(y1_predict-y1)
print('y1总数 = ',np.size(y1),'误差1 =',np.sum(err1),'平均误差',np.sum(err1)/np.size(y1))
err2=np.abs(y2_predict-y2)
print('y2总数 = ',np.size(y2),'误差2 =',np.sum(err2),'平均误差',np.sum(err2)/np.size(y2))

data_test=pd.read_csv('toPredict_Xy.csv')   # 已经排好序，9个9个一组
data_test['time']=pd.to_datetime(data_test['time'])
data_test['time']=[int(x.hour/2) for x in data_test['time']]
data_test=np.array(data_test)
m=np.size(data_test,0)/9
print("m需要是整数，m=",m)
m=int(m)
X_test=np.zeros([m,14])
y0_id_sample=np.zeros(m)
y1_id_sample=np.zeros(m)
y2_id_sample=np.zeros(m)
for i in range(m):
    group=data_test[i*9+0:i*9+9]
    ttl=group[0:6,1]
    id_road=group[0,0]
    speed=group[0:6,2]
    id_sample=group[6:9,4]
    X_test[i]=[id_road,group[3,3],ttl[0],ttl[1],ttl[2],ttl[3],ttl[4],ttl[5],speed[0],speed[1],speed[2],speed[3],speed[4],speed[5]]
    y0_id_sample[i]=group[6,4]
    y1_id_sample[i]=group[7,4]
    y2_id_sample[i]=group[8,4]
# print(X_test[0])
# print(y0_id_sample[0])
# print(y1_id_sample[0])
# print(y2_id_sample[0])
y0_predict=model0.predict(X_test)
y1_predict=model1.predict(X_test)
y2_predict=model2.predict(X_test)
y_predict=np.zeros(m*3)
for i in range(m):
    y_predict[int(y0_id_sample[i])]=y0_predict[i]
    y_predict[int(y1_id_sample[i])]=y1_predict[i]
    y_predict[int(y2_id_sample[i])]=y2_predict[i]
print(y_predict)
out=np.zeros([m*3,2])
for i in range(m*3):
    out[i,0]=i
    out[i,1]=y_predict[i]
print(out)
out=pd.DataFrame(out,columns=['id_sample','TTI'])
out.to_csv('submit3.csv',mode='w',index=False,header=True,encoding= 'utf-8')