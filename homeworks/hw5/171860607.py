from __future__ import print_function
import numpy as np
import pandas as pd

class KNN():
   
    #k: int,最近邻个数.
    def __init__(self, k=5):
        self.k = k

    # 此处需要填写，建议欧式距离，计算一个样本与训练集中所有样本的距离
    def distance(self, one_sample, X_train):
       
        return np.power(np.tile(one_sample,(X_train.shape[0],1))-X_train,2).sum(axis=1)
    
    # 此处需要填写，获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        k_neighbor_labels=[]
        for distance in np.sort(distances)[:k]:
            k_neighbor_labels.extend(list(y_train[distances==distance]))
        return k_neighbor_labels
    # 此处需要填写，标签统计，票数最多的标签即该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train, k):
        distance=self.distance(one_sample,X_train)
        k_neighbor_labels=self.get_k_neighbor_labels(distance,y_train,k)
        labels={}
        for label in k_neighbor_labels:
            if label in labels:
                labels[label]= labels[label]+1
            else:
                labels[label] = 1
        return max(labels, key=lambda x: labels[x])
    
    # 此处需要填写，对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        y_pred=[]
        for one_sample in X_test:
            y_pred.append(self.vote(one_sample,X_train,y_train,self.k))
        return y_pred
  

def main():
    clf = KNN(k=7)
    train_data = np.genfromtxt('./data/train_data.csv', delimiter=' ')
    train_labels = np.genfromtxt('./data/train_labels.csv', delimiter=' ')
    test_data = np.genfromtxt('./data/test_data.csv', delimiter=' ')
   
    #将预测值存入y_pred(list)内    
    y_pred = clf.predict(test_data, train_data, train_labels)
    np.savetxt("171860607_ypred.csv", y_pred, delimiter=' ')


if __name__ == "__main__":
    main()

