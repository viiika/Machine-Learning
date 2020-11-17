import pandas as pd
import matplotlib.pyplot as plt

def problem2_1():
    orgData = pd.read_csv("data.csv")
    # 将数据按照output(预测值)降序重排序
    sortData = orgData.sort_values(by=["output"], ascending=False)
    sortData=sortData.reset_index(drop=True)
    T = 0
    F = 0
    #统计原数据中真实情况的正例和反例各有多少
    for i in range(len(sortData)):
        if sortData["label"][i] == 1:
            T += 1
        else:
            F += 1
    #默认全部预测为反例,计算对应的P值与R值得到点对(P,R)
    TP = 0
    FP = 0
    TN = F
    FN = T
    Plist = [1]
    Rlist = [0]
    #按照预测值的高低,逐个将数据预测为正例,重新计算P与R并保存
    for i in range(len(sortData)):
        if sortData["label"][i] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        #如果发现下一条数据预测值与本条相同,则本条预测值修正为正例后,暂不更新P,R
        if i<len(sortData)-1:
            if sortData["output"][i] == sortData["output"][i+1]:
                continue
        Plist.append(P)
        Rlist.append(R)
    #最终得到保存P和R的两个list,进行绘图
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(Rlist, Plist)
    plt.savefig("P_Rcurve")
    plt.show()

def problem2_2():
    orgData = pd.read_csv("data.csv")
    #将数据按照output(预测值)降序重排序
    sortData = orgData.sort_values(by=["output"], ascending=False)
    sortData = sortData.reset_index(drop=True)
    T = 0
    F = 0
    #统计原数据中真实情况的正例和反例各有多少
    for i in range(len(sortData)):
        if sortData["label"][i] == 1:
            T += 1
        else:
            F += 1
    #默认全部预测为反例,计算对应的TPR值与FPR值得到点对(TPR,FPR)
    TP = 0
    FP = 0
    TN = F
    FN = T
    TPRlist = []
    FPRlist = []
    #按照预测值的高低,逐个将数据预测为正例,重新计算TPR与FPR并保存
    for i in range(len(sortData)):
        if sortData["label"][i] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        #如果发现下一条数据预测值与本条相同,则本条预测值修正为正例后,暂不更新TPR与FPR.
        if i<len(sortData)-1:
            if sortData["output"][i] == sortData["output"][i+1]:
                continue
        TPRlist.append(TPR)
        FPRlist.append(FPR)
    #最终得到保存TPR与FPR的两个list,进行绘图
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(FPRlist, TPRlist)
    plt.savefig("ROCcurve")
    plt.show()
    AUC=0
    for i in range(len(TPRlist)-1):
        AUC+=(0.5*(FPRlist[i+1]-FPRlist[i])*(TPRlist[i]+TPRlist[i+1]))
    print("AUC=",AUC)

if __name__ == '__main__':
    problem2_1()
    problem2_2()