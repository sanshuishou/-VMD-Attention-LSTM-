import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import models.vmd_attention_lstm as mv
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import utilt.VMD as vmdzi
#此处为预测程序

dataframe = pd.read_csv("原数据/处理后的数据表.csv")

K = 3
alpha = 2000
tau = 0.6
vmd = vmdzi.VMD(K, alpha, tau)
hubei_train_set = dataframe.iloc[:1135,4:5].values

other_train_set = dataframe.iloc[:1135,:5].values


#每列作为一个时间步vmd分解 为5*K个特征
# train_vmd = vmdzi.vmd_sp_lis(other_train_set,vmd)
# print(train_vmd.shape)
# # print(train_vmd.shape)
# a = vmdzi.vmd_lis(train_vmd,K)
# np.save('pred_hubei_set.npy',a)
a = np.load('pred_hubei_set.npy')

# 归一化
sc = MinMaxScaler(feature_range=(0 , 1))  # 定义归一化：归一化到(0，1)之间
training_x_set_scaled = sc.fit_transform(a)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
training_y_set_scaled = sc.fit_transform(hubei_train_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化


x_hubei_set = []
y_hubei_set = []



# 测试集：csv表格中前2426-300=2126天数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
for i in range(30, len(training_x_set_scaled)):
    x_hubei_set.append(training_x_set_scaled[i - 30:i, :])
    y_hubei_set.append(training_y_set_scaled[i, :])

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_hubei_set)
np.random.seed(7)
np.random.shuffle(y_hubei_set)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_hubei_set , y_hubei_set  = np.array(x_hubei_set), np.array(y_hubei_set)

y_hubei_set  = np.reshape(y_hubei_set ,(y_hubei_set .shape[0],))

model = mv.attention_lstm(30,K*5,128)


model.load_weights('./checkpoint_/my_model.ckpt')

################## predict ######################
# 测试集输入模型进行预测
predicted_stock_price_t = model.predict(x_hubei_set )

# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price_t = sc.inverse_transform(predicted_stock_price_t[1005:1105])
print(predicted_stock_price_t.shape)

# 对真实数据还原---从（0，1）反归一化到原始范围
y_train = tf.reshape(y_hubei_set ,(-1,1))
real_stock_price_t = sc.inverse_transform(y_train[1005:1105])
print(real_stock_price_t.shape)

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price_t, color='red', label='HuBei Close Price')
plt.plot(predicted_stock_price_t, color='blue', label='Predicted HuBei Close Price')
plt.title('HuBei Close Price traindataset Prediction')
plt.xlabel('Time')
plt.ylabel('HuBei Close Price')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price_t, real_stock_price_t)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price_t, real_stock_price_t))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price_t, real_stock_price_t)
mape = mean_absolute_percentage_error(predicted_stock_price_t, real_stock_price_t)
print('MSE: %.6f' % mse)
print('RMSE: %.6f' % rmse)
print('MAE: %.6f' % mae)
print('MAPE: %.6f' % mape)