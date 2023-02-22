import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import models.vmd_attention_lstm as mv
import utilt.VMD as vmdzi
#此处为训练程序

dataframe = pd.read_csv("原数据/处理后的数据表.csv")

K = 3
alpha = 2000
tau = 0.6
vmd = vmdzi.VMD(K, alpha, tau)
hubei_train_set = dataframe.iloc[:1135-235,4:5].values
hubei_test_set = dataframe.iloc[1135-235:1135,4:5].values

other_train_set = dataframe.iloc[:1135-235,:5].values
other_test_set = dataframe.iloc[1135-235:1135,:5].values


#每列作为一个时间步vmd分解 为5*K个特征

#保存处理后的数据为npy
# train_vmd,test_vmd = vmdzi.vmd_sp_lis(other_train_set,other_test_set,vmd)
# # print(train_vmd.shape)
# a,b = vmdzi.vmd_lis(train_vmd,test_vmd,K)
# # print(a.shape)
# np.save('train_vmd_af.npy',a)
# np.save('test_vmd_af.npy',b)

#加载已保存的npy数据
a,b = np.load('train_vmd_af.npy'),np.load('test_vmd_af.npy')

# 归一化
sc = MinMaxScaler(feature_range=(0 , 1))  # 定义归一化：归一化到(0，1)之间
training_x_set_scaled = sc.fit_transform(a)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
training_y_set_scaled = sc.fit_transform(hubei_train_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_x_set = sc.fit_transform(b)  # 利用训练集的属性对测试集进行归一化
test_y_set = sc.fit_transform(hubei_test_set)  # 利用训练集的属性对测试集进行归一化



x_train = []
y_train = []

x_test = []
y_test = []

# 利用for循环，遍历整个训练集，提取训练集中连续30天的收盘价作为输入特征x_train，第30天的数据作为标签，for循环共构建1135-235-60=840组数据。
for i in range(30, len(training_x_set_scaled)):
    x_train.append(training_x_set_scaled[i - 30:i, :])
    y_train.append(training_y_set_scaled[i, :])

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)

y_train = np.reshape(y_train,(y_train.shape[0],))


# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。

# 利用for循环，遍历整个测试集，提取测试集中连续30天的收盘价作为输入特征x_train，第31天的数据作为标签，for循环共构建235-30=200组数据。
for i in range(30, len(test_x_set)):
    x_test.append(test_x_set[i - 30:i, :])
    y_test.append(test_y_set[i, :])
# 测试集变array并reshape为符合LSTM输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
y_test = np.reshape(y_test,(y_test.shape[0],)) #标签为一维数组


model = mv.attention_lstm(30,K*5,128) #attention_lstm(时间步长，特征数量，lstm神经元数量)

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(1e-4))

histroy = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=3000,batch_size=128,callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoint_/my_model.ckpt',save_weights_only=True,save_best_only=True)])

