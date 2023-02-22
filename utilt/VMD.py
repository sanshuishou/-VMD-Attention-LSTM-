from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import hilbert
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#vmd分解类
class VMD:
    def __init__(self, K, alpha, tau, tol=1e-7, maxIters=200, eps=1e-9):
        """
        :param K: 模态数
        :param alpha: 每个模态初始中心约束强度
        :param tau: 对偶项的梯度下降学习率
        :param tol: 终止阈值
        :param maxIters: 最大迭代次数
        :param eps: eps
        """
        self.K =K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.maxIters = maxIters
        self.eps = eps

    def __call__(self, f):
        N = f.shape[0]
        # 对称拼接
        f = np.concatenate((f[:N // 2][::-1], f, f[N // 2:][::-1]))
        T = f.shape[0]
        t = np.linspace(1, T, T) / T
        omega = t - 1. / T
        # 转换为解析信号
        f = hilbert(f)
        f_hat = np.fft.fft(f)
        u_hat = np.zeros((self.K, T), dtype=np.complex128)
        omega_K = np.zeros((self.K,))
        lambda_hat = np.zeros((T,), dtype=np.complex128)
        # 用以判断
        u_hat_pre = np.zeros((self.K, T), dtype=np.complex128)
        u_D = self.tol + self.eps

        # 迭代
        n = 0
        while n < self.maxIters and u_D > self.tol:
            for k in range(self.K):
                # u_hat
                sum_u_hat = np.sum(u_hat, axis=0) - u_hat[k, :]
                res = f_hat - sum_u_hat
                u_hat[k, :] = (res + lambda_hat / 2) / (1 + self.alpha * (omega - omega_K[k]) ** 2)

                # omega
                u_hat_k_2 = np.abs(u_hat[k, :]) ** 2
                omega_K[k] = np.sum(omega * u_hat_k_2) / np.sum(u_hat_k_2)

            # lambda_hat
            sum_u_hat = np.sum(u_hat, axis=0)
            res = f_hat - sum_u_hat
            lambda_hat -= self.tau * res

            n += 1
            u_D = np.sum(np.abs(u_hat - u_hat_pre) ** 2)
            u_hat_pre[::] = u_hat[::]

            # 重构，反傅立叶之后取实部
        u = np.real(np.fft.ifft(u_hat, axis=-1))
        u = u[:, N // 2: N // 2 + N]

        omega_K = omega_K * T / 2
        idx = np.argsort(omega_K)
        omega_K = omega_K[idx]
        u = u[idx, :]
        return u, omega_K

#重特征数组中将每个数据点与特征一一对应
def vmd_lis(train_set_vmd,test_set_vmd,input_K):
    vmd_train_ls = []
    vmd_test_ls = []
    vmd_train_ls_1 = []
    vmd_test_ls_1 = []
    for i in range(train_set_vmd.shape[0]):
        for j in range(train_set_vmd.shape[2]):
            for k in range(train_set_vmd.shape[1]):
                vmd_train_ls_1.append(train_set_vmd[i][k][j])
        if len(vmd_train_ls_1) == 5*input_K :
            vmd_train_ls.append(vmd_train_ls_1)
            vmd_train_ls_1 = []
    for i in range(test_set_vmd.shape[0]):
        for j in range(test_set_vmd.shape[2]):
            for k in range(test_set_vmd.shape[1]):
                vmd_test_ls_1.append(test_set_vmd[i][k][j])
        if len(vmd_test_ls_1) == 5*input_K :
            vmd_test_ls.append(vmd_test_ls_1)
            vmd_test_ls_1 = []
    return np.array(vmd_train_ls),np.array(vmd_test_ls)

#将数据集vmd分解且分为训练集与测试集的特征数组
def vmd_sp_lis(other_train_set,other_test_set,vmd):
    vmd_train_lis = []
    vmd_test_lis = []
    for i in range(len(other_train_set)):
        kk,_ = vmd(other_train_set[i])
        vmd_train_lis.append(kk)
    for j in range(len(other_test_set)):
        kk,_ = vmd(other_test_set[j])
        vmd_test_lis.append(kk)
    return np.array(vmd_train_lis),np.array(vmd_test_lis)

#vmd测试
# if __name__ == '__main__':
#     K = 12
#     alpha = 2000
#     tau = 1e-6
#     vmd = VMD(K, alpha, tau)
#     dataframe = pd.read_csv("../zgpa_train.csv")
#     dataset = dataframe.iloc[:,5:].values
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     dataset = scaler.fit_transform(dataset.reshape(-1, 1))
#
#     train_size = int(len(dataset)*0.8)
#     test_size = len(dataset)-train_size
#     train, test = dataset[0: train_size], dataset[train_size: len(dataset)]
#
#     look_back = 1
#     trainX, trainY = dt.creat_dataset(train, look_back)
#     testX, testY = dt.creat_dataset(test, look_back)
#     data = np.reshape(trainX,(trainX.shape[0],))
#     u, omega_K = vmd(data)
#     print(omega_K)
#     jj
#     # array([  9.68579292,  50.05232833, 100.12321047])
#     print(u.shape)
#     plt.figure(figsize=(5,7), dpi=200)
#     plt.subplot(3,1,1)
#     # plt.plot(mode_1, linewidth=0.5, linestyle='--')
#     plt.plot(u[0,:], linewidth=0.2, c='r')
#
#     plt.subplot(3,1,2)
#     # plt.plot(mode_2, linewidth=0.5, linestyle='--')
#     plt.plot(u[1,:], linewidth=0.2, c='r')
#
#     plt.subplot(3,1,3)
#     # plt.plot(mode_3, linewidth=0.5, linestyle='--')
#     plt.plot(u[2,:], linewidth=0.2, c='r')
#
#     plt.show()