import pandas as pd
import numpy as np
import random
from cvxopt import matrix, solvers


def load_data():
    X_train = pd.read_csv("X_train.csv", header=None)
    Y_train = pd.read_csv("y_train.csv", header=None)
    X_test = pd.read_csv("X_test.csv", header=None)

    X_train = X_train.values
    Y_train = Y_train.values.reshape((1, len(X_train)))
    Y_train[Y_train == 0] = -1
    Y_train = Y_train.astype(float)[0]

    X_test = X_test.values

    return X_train, Y_train, X_test

def load_data_for_validation():
    X_train = pd.read_csv("X_train.csv", header=None)
    Y_train = pd.read_csv("y_train.csv", header=None)

    train = pd.concat([X_train, Y_train], axis=1,ignore_index=True)
    train = train.sample(frac=1.0)
    X_train = train.iloc[:,:5]
    Y_train = train.iloc[:,5:]

    X_valid = X_train.iloc[280:]
    Y_valid = Y_train.iloc[280:]
    X_train = X_train.iloc[:280]
    Y_train = Y_train.iloc[:280]

    X_train = X_train.values
    Y_train = Y_train.values.reshape((1, len(X_train)))
    Y_train[Y_train == 0] = -1
    Y_train = Y_train.astype(float)[0]

    X_valid = X_valid.values
    Y_valid = Y_valid.values.reshape((1, len(X_valid)))
    Y_valid[Y_valid == 0] = -1
    Y_valid = Y_valid.astype(float)[0]
    return X_train, Y_train, X_valid, Y_valid

def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def poly_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def RBF_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def qp(X_train, Y_train, C=None, kernel=RBF_kernel):
    samples, features = X_train.shape

    K = np.zeros((samples, samples))
    for i in range(samples):
        for j in range(samples):
            K[i, j] = kernel(X_train[i], X_train[j])

    P = matrix(np.outer(Y_train, Y_train) * K)
    q = matrix(np.ones(samples) * -1)
    A = matrix(Y_train, (1, samples))
    b = matrix(0.0)
    if C is None:
        G = matrix(np.diag(np.ones(samples) * -1))
        h = matrix(np.zeros(samples))
    else:
        G = matrix(np.vstack((np.diag(np.ones(samples) * -1), np.identity(samples))))
        h = matrix(np.hstack((np.zeros(samples), np.ones(samples) * C)))
    solution = solvers.qp(P, q, G, h, A, b)

    return solution


def smo(X_train, Y_train, C=None, toler=0.0001, maxIter=1000):
    # ref:http://cs229.stanford.edu/materials/smo.pdf
    samples, features = X_train.shape
    X_train = np.matrix(X_train)
    Y_train = np.matrix(Y_train).T
    b = 0.0
    alpha = np.matrix(np.zeros((samples, 1)))
    curIter = 0
    while curIter < maxIter:
        num_changed_alphas = 0
        for i in range(samples):
            fXi = np.multiply(alpha, Y_train).T * (X_train * X_train[i, :].T) + b
            Ei = fXi - Y_train[i]
            if Y_train[i] * Ei < -toler and alpha[i] < C or Y_train[i] * Ei > toler and alpha[i] > 0:
                j = i
                while j == i:
                    j = int(random.uniform(0, samples))
                fXj = np.multiply(alpha, Y_train).T * (X_train * X_train[j, :].T) + b

                Ej = fXj - Y_train[j]
                last_alpha_I = alpha[i].copy()
                last_alpha_J = alpha[j].copy()
                if Y_train[i] != Y_train[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    continue
                eta = 2.0 * X_train[i, :] * X_train[j, :].T - X_train[i, :] * X_train[i, :].T - X_train[j, :] * X_train[j, :].T
                if eta >= 0:
                    continue
                alpha[j] -= Y_train[j] * (Ei - Ej) / eta
                if alpha[j] > H:
                    alpha[j] = H
                if alpha[j] < L:
                    alpha[j] = L
                if abs(alpha[j] - last_alpha_J) < 1e-5:
                    continue
                alpha[i] += Y_train[j] * Y_train[i] * (last_alpha_J - alpha[j])  # 调整alphas[i]
                b1 = b - Ei - Y_train[i] * (alpha[i] - last_alpha_I) * X_train[i, :] * X_train[i, :].T - Y_train[j] * (alpha[j] - last_alpha_J) * X_train[i, :] * X_train[j, :].T
                b2 = b - Ej - Y_train[i] * (alpha[i] - last_alpha_I) * X_train[i, :] * X_train[j, :].T - Y_train[j] * (alpha[j] - last_alpha_J) * X_train[j, :] * X_train[j, :].T

                if 0 < alpha[i] and C > alpha[i]:
                    b = b1
                elif 0 < alpha[j] and C > alpha[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            curIter += 1
        else:
            curIter = 0

    return b, alpha

def predict(X_test, w=None, b=None, a=None, sv=None, sv_y=None, kernel=RBF_kernel):
    if w is None:
        Y_predict_float = np.zeros(len(X_test))
        for i in range(len(X_test)):
            s = 0
            for aj, sv_yj, svj in zip(a, sv_y, sv):
                s += aj * sv_yj * kernel(X_test[i], svj)
            Y_predict_float[i] = s
        Y_predict = Y_predict_float + b
        return np.sign(Y_predict)
    else:  # linear_kernel
        Y_predict = np.dot(X_test, w) + b
        return np.sign(np.squeeze(Y_predict.ravel())).tolist()[0]


def qp_validation(X_train, Y_train, C=None, kernel=RBF_kernel):
    samples, features = X_train.shape
    K = np.zeros((samples, samples))
    for i in range(samples):
        for j in range(samples):
            K[i, j] = kernel(X_train[i], X_train[j])
    solution = qp(X_train, Y_train, C, RBF_kernel)
    solution_x = np.ravel(solution['x'])
    sv_index = solution_x > 1e-5
    index = np.arange(len(solution_x))[sv_index]
    a = solution_x[sv_index]  # support_vectors
    sv = X_train[sv_index]
    sv_y = Y_train[sv_index]

    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[index[n], sv_index])
    b /= len(a)

    if kernel == linear_kernel:
        w = np.zeros(features)
        for n in range(len(a)):
            w += a[n] * sv_y[n] * sv[n]
    else:
        w = None

    return w, b, a, sv, sv_y


def smo_validation(X_train, Y_train, C=None, toler=0.0001, maxIter=1000):
    b, alpha = smo(X_train, Y_train, C, toler, maxIter)
    alpha, X_train, Y_train = np.array(alpha), np.array(X_train), np.array(Y_train)
    w = np.dot((np.tile(Y_train.reshape(1, -1).T, (1, 5)) * X_train).T, alpha).tolist()
    return w, b

def train_and_find_better_paras():
    # 需要调整哪个参数，外面直接加一个大循环，设置好初始点、步长、终止点，等待输出即可
    X_train, Y_train, X_valid, Y_valid= load_data_for_validation()
    w_qp, b_qp, a, sv, sv_y = qp_validation(X_train, Y_train, C=1)
    Y_predict_qp = predict(X_valid, w_qp, b_qp, a, sv, sv_y)

    w_smo,b_smo = smo_validation(X_train, Y_train, 1, 0.001, 100)
    Y_predict_smo = predict(X_valid, w_smo, b_smo)

    correct = np.sum(Y_predict_qp == Y_valid)
    print("%d out of %d predictions correct" % (correct, len(Y_predict_qp)))
    correct = np.sum(Y_predict_smo == Y_valid)
    print("%d out of %d predictions correct" % (correct, len(Y_predict_smo)))


if __name__ == '__main__':
    X_train, Y_train , X_test = load_data()

    # 第一问
    # solution = qp(X_train, Y_train, C=0.1)
    # 第二问
    # w, alpha = smo(X_train, Y_train, 1, 0.001, 10)
    # 第三问
    train_and_find_better_paras()

    w_smo, b_smo = smo_validation(X_train, Y_train, 1, 0.0001, 1)
    Y_predict_smo = predict(X_test, w_smo, b_smo)
    Y_predict_smo = np.array(Y_predict_smo, dtype=np.int)
    Y_predict_smo[Y_predict_smo == -1] = 0
    output = pd.DataFrame(Y_predict_smo)
    output.to_csv("171860607.csv", index=False, header=None, sep=',')

