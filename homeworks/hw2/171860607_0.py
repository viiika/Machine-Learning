import pandas as pd
import numpy as np
import math


def loading(X_train_path,Y_train_path,X_val_path,Y_val_path):
    X_train = pd.read_csv(X_train_path)
    Y_train = pd.read_csv(Y_train_path)
    X_train['add_column'] = 1
    X_val = pd.read_csv(X_val_path)
    Y_val = pd.read_csv(Y_val_path)
    X_val['add_column'] = 1
    return X_train,Y_train,X_val,Y_val


def answer1(fixedbar,z):
    Y_guess = [1 / (1 + math.pow(math.e, -z[i])) for i in range(len(z))]
    bar = fixedbar
    Y_guess = np.array(Y_guess)
    Y_guess = np.where(Y_guess > bar, 1, 0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(Y_val)):
        if Y_val.values[i] == 1 and Y_guess[i] == 1:
            TP += 1
        elif Y_val.values[i] == 1 and Y_guess[i] == 0:
            FN += 1
        elif Y_val.values[i] == 0 and Y_guess[i] == 1:
            FP += 1
        elif Y_val.values[i] == 0 and Y_guess[i] == 0:
            TN += 1
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    print("Accuracy=", Accuracy)
    print("Precision=", Precision)
    print("Recall=", Recall)
    #输出为
    #Accuracy= 0.74
    #Precision= 0.6666666666666666
    #Recall= 1.0

def train(X_train,Y_train,X_val,Y_val,bar,step):
    w = np.dot(
            np.dot(
                np.linalg.pinv(np.dot(np.transpose(X_train.values), X_train.values)),
                np.transpose(X_train.values)),
            Y_train.values)

    z = np.dot(X_val.values, w)
    Y_guess_ = [1 / (1 + math.pow(math.e, -z[i])) for i in range(len(z))]

    answer1(fixedbar=0.5,z=z) #第一题的解答

    #接下来是寻找效果最好的bar
    Accuracy = 0
    MaxAcc = 0
    MaxAccBar = 0
    while abs(Accuracy-1)>1e-2 and abs(bar-1)>1e-2:#从bar开始，step大小递增，如果acc=1就跳出，否则一直找更大的
        Y_guess = np.array(Y_guess_)
        Y_guess = np.where(Y_guess > bar, 1, 0)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(Y_val)):
            if Y_val.values[i] == 1 and Y_guess[i] == 1:
                TP += 1
            elif Y_val.values[i] == 1 and Y_guess[i] == 0:
                FN += 1
            elif Y_val.values[i] == 0 and Y_guess[i] == 1:
                FP += 1
            elif Y_val.values[i] == 0 and Y_guess[i] == 0:
                TN += 1
        #Precision = TP / (TP + FP)
        #Recall = TP / (TP + FN)
        Accuracy = (TP + TN) / (TP + FP + TN + FN)
        if Accuracy>MaxAcc:
            MaxAcc=Accuracy
            MaxAccBar=bar
        bar+=step

    print("MaxAccBar:",MaxAccBar)
    print("MaxAcc",MaxAcc)
    return w,MaxAccBar

def predict(w,bar,X_test_path):
    X_test = pd.read_csv(X_test_path)
    X_test['add_column'] = 1
    z = np.dot(X_test.values, w)
    Y_guess = [1 / (1 + math.pow(math.e, -z[i])) for i in range(len(z))]
    Y_guess = np.array(Y_guess)
    Y_guess = np.where(Y_guess > bar, 1, 0)
    Y_guess = pd.DataFrame(Y_guess)
    Y_guess.to_csv("171860607_0.csv", index=False, sep=',')
    return Y_guess


if __name__ == '__main__':
    X_train_path="/Users/bryan/PycharmProjects/ProJ1/ML_HW2/train_feature.csv"
    Y_train_path="/Users/bryan/PycharmProjects/ProJ1/ML_HW2/train_target.csv"
    X_val_path="/Users/bryan/PycharmProjects/ProJ1/ML_HW2/val_feature.csv"
    Y_val_path="/Users/bryan/PycharmProjects/ProJ1/ML_HW2/val_target.csv"
    X_train,Y_train,X_val,Y_val=loading(X_train_path,Y_train_path,X_val_path,Y_val_path)
    w,MaxAccBar=train(X_train,Y_train,X_val,Y_val,bar=0,step=0.01)#bar起始阈值，step步长
    X_test_path="/Users/bryan/PycharmProjects/ProJ1/ML_HW2/test_feature.csv"
    predict(w,bar=MaxAccBar,X_test_path=X_test_path)