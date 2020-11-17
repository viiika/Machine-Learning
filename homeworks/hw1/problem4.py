import math
import matplotlib.pyplot as plt

def Friedman(performance,avg):
    N = len(performance)
    k = len(performance[0])
    a = (12 * N) / (k * (k + 1))
    sum = 0
    for i in avg:
        sum += (i * i)
    sum -= (k * (k + 1) * (k + 1) / 4)
    Tx2 = a * sum
    TF = ((N - 1) * Tx2) / (N * (k - 1) - Tx2)
    print("Friedman:",TF)

def Nemenyi(performance,avg):
    N = len(performance)
    k = len(performance[0])
    qa=2.728
    CD=qa*math.sqrt((k*(k+1))/(6*N))
    print("Nemenyi:",CD)
    return CD

if __name__ == '__main__':
    performance = [[2, 3, 1, 5, 4],
                   [5, 4, 2, 3, 1],
                   [4, 5, 1, 2, 3],
                   [2, 3, 1, 5, 4],
                   [3, 4, 1, 5, 2]]
    avg = [3.2, 3.8, 1.2, 4, 2.8]
    avg_name=["alg_A","alg_B","alg_C","alg_D","alg_E"]
    Friedman(performance, avg)
    CD=Nemenyi(performance,avg)
    plt.scatter(avg,avg_name)
    algmin=[]
    algmax=[]
    for i in avg:
        algmin.append(i-CD/2)
        algmax.append(i+CD/2)
    plt.hlines(avg_name,algmin,algmax)
    plt.savefig("Friedmanfig")
    plt.show()