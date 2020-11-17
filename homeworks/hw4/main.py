import numpy as np
# sigmoid
def sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)）
    return 1.0 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度）
    return sigmoid(x) * sigmoid(1-x)


# loss
def mse_loss(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差）,其中真实标记和预测值都是长度相同的向量）
    return np.sum(np.square(y_true - y_pred))/len(y_true)


class NeuralNetwork_advanced():
    def __init__(self,layers_strcuture):
        self.layers_strcuture = layers_strcuture
        self.layers_num = len(layers_strcuture)
        # 去掉输入层即为隐藏层数
        self.param_layers_num = self.layers_num - 1
        # weights
        self.w = dict()
        # biases
        self.b = dict()
        # 以上为神经网络中的变量，其中具体含义见网络图
        np.random.seed(3)
        for l in range(1, self.layers_num):
            self.w["w" + str(l)] = np.random.randn(self.layers_strcuture[l], self.layers_strcuture[l - 1]) / np.sqrt(
                self.layers_strcuture[l - 1])
            self.b["b" + str(l)] = np.zeros((self.layers_strcuture[l], 1))

    def predict(self,x):
        output_prev = x
        L = self.param_layers_num
        for l in range(1, L):
            input_cur = output_prev
            output_prev = sigmoid(np.dot(self.w["w" + str(l)], input_cur) + self.b["b" + str(l)])

        input_cur = output_prev

        output = sigmoid(np.dot(self.w["w" + str(L)], input_cur) + self.b["b" + str(L)])
        output = output.T
        result = output / np.sum(output, axis=1, keepdims=True)
        return np.argmax(result, axis=1)

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 30000
        for epoch in range(epochs):
            caches = []
            output_prev = data
            # 向前传播
            L = self.param_layers_num
            for l in range(1, L):
                input_cur = output_prev
                output_prev = sigmoid(np.dot(self.w["w" + str(l)], input_cur) + self.b["b" + str(l)])
                caches.append((input_cur, self.w["w" + str(l)], self.b["b" + str(l)],
                               np.dot(self.w["w" + str(l)], input_cur) + self.b["b" + str(l)]))
            input_cur = output_prev
            output_prev = sigmoid(np.dot(self.w["w" + str(L)], input_cur) + self.b["b" + str(L)])
            caches.append((input_cur, self.w["w" + str(L)], self.b["b" + str(L)],
                           np.dot(self.w["w" + str(L)], input_cur) + self.b["b" + str(L)]))

            m = all_y_trues.shape[1]
            error = -np.sum(np.multiply(np.log(output_prev), all_y_trues) + np.multiply(np.log(1 - output_prev),
                                                                                        1 - all_y_trues)) / m
            loss = np.squeeze(error)

            # 后向传播
            grads = {}
            L = self.param_layers_num  #
            output_prev = output_prev.reshape(output_prev.shape)
            expected_output = all_y_trues

            # 交叉熵作为误差函数
            derror_wrt_output = - (np.divide(expected_output, output_prev) - np.divide(1 - expected_output, 1 - output_prev))

            input, w, b, input_sum = caches[L - 1]
            output_prev = input
            m = output_prev.shape[1]
            derror_wrt_dinput = derror_wrt_output*deriv_sigmoid(input_sum)
            grads["derror_wrt_dw" + str(L)] = np.dot(derror_wrt_dinput, output_prev.T) / m
            grads["derror_wrt_db" + str(L)] = np.sum(derror_wrt_dinput, axis=1, keepdims=True) / m
            grads["derror_wrt_output" + str(L)] = np.dot(w.T, derror_wrt_dinput)
            for l in reversed(range(L - 1)):
                input, w, b, input_sum = caches[l]
                output_prev = input
                m = output_prev.shape[1]
                derror_wrt_dinput = grads["derror_wrt_output" + str(l + 2)]*deriv_sigmoid(input_sum)
                grads["derror_wrt_dw" + str(l+1)] = np.dot(derror_wrt_dinput, output_prev.T) / m
                grads["derror_wrt_db" + str(l+1)] = np.sum(derror_wrt_dinput, axis=1, keepdims=True) / m
                grads["derror_wrt_output" + str(l+1)] = np.dot(w.T, derror_wrt_dinput)

            # 更新权重和偏置
            for l in range(self.param_layers_num):
                self.w["w" + str(l + 1)] = self.w["w" + str(l + 1)] - learn_rate * grads[
                    "derror_wrt_dw" + str(l + 1)]
                self.b["b" + str(l + 1)] = self.b["b" + str(l + 1)] - learn_rate * grads[
                    "derror_wrt_db" + str(l + 1)]

            if epoch % 100 == 0:
                print("Epoch %d loss: %.3f" % (epoch, loss))

class NeuralNetwork_221():
    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        # 以上为神经网络中的变量，其中具体含义见网络图

    def predict(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 3000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 以下部分为向前传播过程，请完成
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1 # （需要填写的地方，含义为隐层第一个节点收到的输入之和）
                h1 =  sigmoid(sum_h1)# （需要填写的地方，含义为隐层第一个节点的输出）

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2 # （需要填写的地方，含义为隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2) # （需要填写的地方，含义为隐层第二个节点的输出）

                sum_ol = self.w5 * h1 + self.w6 * h2 + self.b3 # （需要填写的地方，含义为输出层节点收到的输入之和）
                ol = sigmoid(sum_ol) # （需要填写的地方，含义为输出层节点的对率输出）
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred = -2*(y_true-y_pred) # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = h1*deriv_sigmoid(sum_ol) # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_ypred_d_w6 = h2*deriv_sigmoid(sum_ol) # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_ypred_d_b3 = deriv_sigmoid(sum_ol) # （需要填写的地方，含义为输出层对率输出对b3的梯度）
                d_ypred_d_h1 = self.w5*deriv_sigmoid(sum_ol) # （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = self.w6*deriv_sigmoid(sum_ol) # （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 隐层梯度
                d_h1_d_w1 =  x[0]*deriv_sigmoid(sum_h1)# （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 =  x[1]*deriv_sigmoid(sum_h1)# （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 =  deriv_sigmoid(sum_h1)# （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 =  x[0]*deriv_sigmoid(sum_h2)# （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 =  x[1]*deriv_sigmoid(sum_h2)# （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 =  deriv_sigmoid(sum_h2)# （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w5 -= d_L_d_ypred*d_ypred_d_w5*learn_rate # （需要填写的地方，更新w5）
                self.w6 -= d_L_d_ypred*d_ypred_d_w6*learn_rate # （需要填写的地方，更新w6）
                self.b3 -= d_L_d_ypred*d_ypred_d_b3*learn_rate # （需要填写的地方，更新b3）

                self.w1 -= d_L_d_ypred*d_ypred_d_h1*d_h1_d_w1*learn_rate # （需要填写的地方，更新w1）
                self.w2 -= d_L_d_ypred*d_ypred_d_h1*d_h1_d_w2*learn_rate # （需要填写的地方，更新w2）
                self.b1 -= d_L_d_ypred*d_ypred_d_h1*d_h1_d_b1*learn_rate # （需要填写的地方，更新b1）

                self.w3 -= d_L_d_ypred*d_ypred_d_h2*d_h2_d_w3*learn_rate # （需要填写的地方，更新w3）
                self.w4 -= d_L_d_ypred*d_ypred_d_h2*d_h2_d_w4*learn_rate# （需要填写的地方，更新w4）
                self.b2 -= d_L_d_ypred*d_ypred_d_h2*d_h2_d_b2*learn_rate # （需要填写的地方，更新b2）

            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f"% (epoch, loss))


def advanced_main():
    import numpy as np
    X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
    y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')
    # 打乱训练集顺序
    # permutation = np.random.permutation(y_train.shape[0])
    # X_train = X_train[permutation]
    # y_train = y_train[permutation]
    # X_valid=X_train[300:]
    # y_valid=y_train[300:]
    X_train = X_train[:300]
    y_train = y_train[:300]
    X_test = np.genfromtxt('./data/test_feature.csv', delimiter=',')
    expect_output = []
    for c in y_train:
        if c == 1:
            expect_output.append([0, 1])
        else:
            expect_output.append([1, 0])
    expect_output = np.array(expect_output).T
    network = NeuralNetwork_advanced([2,2, 2])
    network.train(X_train.T, expect_output)

    # y_pred = network.predict(X_valid.T)
    # print(1-sum(abs(y_pred - y_valid))/100)


    y_pred=network.predict(X_test.T)

    # with open('171860607_ypred.csv', 'w') as ypred:
    #     for i in y_pred:
    #         ypred.write(str(i)+'\n')

def main():
    import numpy as np
    X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
    y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')
    X_test = np.genfromtxt('./data/test_feature.csv', delimiter=',')#读取测试样本特征
    network = NeuralNetwork_221()
    network.train(X_train, y_train)

    y_pred=[]
    for i in X_test:
        y_pred.append(network.predict(i))#将预测值存入y_pred(list)内
    ##############
    # （需要填写的地方，选定阈值，将输出对率结果转化为预测结果并输出）
    y_out=[]
    bar=0.5
    for i in y_pred:
        if i>bar:
            y_out.append(1)
        else:
            y_out.append(0)
    # print(y_out)
    # with open('final171860607_ypred.csv','w') as ypred:
    #      for i in y_out:
    #          ypred.write(str(i)+'\n')
    ##############

#main()
advanced_main()
