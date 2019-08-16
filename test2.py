# -*- coding: utf-8 -*-
import xlrd
import xlwt
import random
import numpy as np
import tensorflow as tf
from sklearn import svm
import os
import math



np.set_printoptions(threshold=np.inf)




sess = tf.InteractiveSession()


# 建立一个tensorflow的会话

# 初始化权值向量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


    # 初始化偏置向量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 二维卷积运算，步长为1，输出大小不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    # 池化运算，将卷积特征缩小为1/2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def doubleS1(x):
    output = 10*(2+math.exp(10*(2.3-x)))/(1+math.exp(10*(-2.3-x)))/(1+math.exp(10*(2.3-x)))
    return output

def doubleS2(x):
    output = 10*(2+math.exp(10*(4.3-x)))/(1+math.exp(10*(-4.3-x)))/(1+math.exp(10*(4.3-x)))
    return output

def doubleS3(x):
    output = 10*(2+math.exp(10*(1.8-x)))/(1+math.exp(10*(-1.8-x)))/(1+math.exp(10*(1.8-x)))
    return output

def cnn_train():
    batch = ([], [])
    row = sheet1.row_values(i)
    # print(row)
    minus=[10, 18, 18, 4, 4, 4]
    x_test2=[0, 0, 0, 0, 0, 0]
    #print(row)
    for l in range(len(row)):
        x_test2[l] = row[l] - minus[l]
    x_test2[0] = doubleS1(x_test2[0])
    x_test2[1] = doubleS2(x_test2[1])
    x_test2[2] = doubleS2(x_test2[2])
    x_test2[3] = doubleS3(x_test2[3])
    x_test2[4] = doubleS3(x_test2[4])
    x_test2[5] = doubleS3(x_test2[5])
    print(x_test2)
    x_test2 = np.array(x_test2, dtype='float32')
    x_test2 = x_test2.reshape(1, 6)
    #x_test2 = np.linspace(-1, 1, 300)[:, np.newaxis].astype('float32')
    #print(x_test2)
    batch[0].append(x_test2)
    #print(b)
    #y = tf.matmul(x_test2, W) + b
    #x_image = tf.nn.softmax(tf.reshape(y, [-1, 6, 6, 1]))

    # 图片大小是16*16，,-1代表其他维数自适应
    #output = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    #output = conv2d(x_image, w_conv1) + b_conv1
    output=sess.run(y, feed_dict={x: x_test2})
    output = output[0]
    output = output.reshape(-1, 6)
    output = tf.nn.softmax(tf.matmul(output, w_fc2) + b_fc2)
    output = output.eval(session=sess)
    #h_pool1 = output
    # 此时输出的维数是256维
    #h_pool2_flat = tf.reshape(h_pool1, [-1, 4])
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    #print (output)
    #os.system("pause")
    #output = output.eval(session=sess)



    #print(output)
    output = output[0]
    vector=output
    vector=vector.tolist()
    print(vector)
    for j in range(0, len(vector)):
        sheet4.write(k, j, vector[j], style)
    #print(vector)
    #print('-----------------------------------------------')
    return 0



# 给x，y留出占位符，以便未来填充数据
x = tf.placeholder("float", [None, 6])
y_ = tf.placeholder("float", [None, 6])
# 设置输入层的W和b
W = weight_variable([6, 6])
b = bias_variable([6])
# 计算输出，采用的函数是softmax（输入的时候是one hot编码）

y = tf.sigmoid(tf.matmul(x, W) + b)

# 设置dropout，否则很容易过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(y, keep_prob)

# 输出层，在本实验中只利用它的输出反向训练CNN，至于其具体数值我不关心
w_fc2 = weight_variable([6, 6])
b_fc2 = bias_variable([6])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#square_loss = -tf.multiply((y_-y_conv), (y_-y_conv))
# 设置误差代价以交叉熵的形式
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(square_loss)
# 用adma的优化算法优化目标函数
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

ExcelFile=xlrd.open_workbook(r'C:\Users\lenovo\Desktop\data.xlsx')

#获取目标EXCEL文件sheet名

print (ExcelFile.sheet_names())
sheet1=ExcelFile.sheet_by_index(0)
sheet2=ExcelFile.sheet_by_index(1)

for i in random.sample(range(17980), 17980):
#for i in p:
    batch = ([], [])
    row=sheet1.row_values(i)

    minus=[10, 18, 18, 4, 4, 4]
    x_test2=[0, 0, 0, 0, 0, 0]
    #print(row)
    for l in range(len(row)):
        x_test2[l] = row[l] - minus[l]
    x_test2[0] = doubleS1(x_test2[0])
    x_test2[1] = doubleS2(x_test2[1])
    x_test2[2] = doubleS2(x_test2[2])
    x_test2[3] = doubleS3(x_test2[3])
    x_test2[4] = doubleS3(x_test2[4])
    x_test2[5] = doubleS3(x_test2[5])
    print (x_test2)
    x_test2 = np.array(x_test2)
    #x_test2 = x_test2.reshape(1, 6)
    batch[0].append(x_test2)
    #print (type(x_test2))
    #x_test2 = tf.reshape(x_test2, [-1, 1, 6, 1])
    #x_test2 = x_test2.eval()
    #print(x_test2)
    y_test2=[]
    flag=sheet2.row_values(i)
    #print (flag)
    if flag == [1]:
        file_num='1'
        ff2 = open('posi' + file_num + '.data')
        rr2 = ff2.readline()
        y_test2.append(list(map(float, rr2.split())))
    elif flag==[2]:
        file_num='2'
        ff2 = open('posi' + file_num + '.data')
        rr2 = ff2.readline()
        y_test2.append(list(map(float, rr2.split())))
    elif flag==[4]:
        file_num='3'
        ff2 = open('posi' + file_num + '.data')
        rr2 = ff2.readline()
        y_test2.append(list(map(float, rr2.split())))
    elif flag==[8]:
        file_num='4'
        ff2 = open('posi' + file_num + '.data')
        rr2 = ff2.readline()
        y_test2.append(list(map(float, rr2.split())))
    elif flag==[16]:
        file_num='5'
        ff2 = open('posi' + file_num + '.data')
        rr2 = ff2.readline()
        y_test2.append(list(map(float, rr2.split())))
    else:
        file_num='6'
        ff2 = open('posi' + file_num + '.data')
        rr2 = ff2.readline()
        y_test2.append(list(map(float, rr2.split())))
    y_test2 = y_test2[0]
    #print (y_test2)
    y_test2 = np.array(y_test2)
    batch[1].append(y_test2)
    #print(y_test2)
    #print(batch[1])
    #print(i)
    #print(batch[0])
    #print(batch[1])
    for time in range(5):
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})

f = xlwt.Workbook()
sheet4 = f.add_sheet('Sheet4', cell_overwrite_ok=True)



#q = random.sample(range(17980), 5)
style = set_style('Times New Roman', 220, True)
k = 0
for i in range(0, 300):
    cnn_train()
    k = k+1
for i in range(960, 1260):
    cnn_train()
    k = k + 1
for i in range(2300, 2600):
    cnn_train()
    k = k + 1
for i in range(9900, 10200):
    cnn_train()
    k = k + 1
for i in range(15200, 15500):
    cnn_train()
    k = k + 1
for i in range(15900, 16200):
    cnn_train()
    k = k + 1
f.save(r'C:\Users\lenovo\Desktop\datatezheng1.xlsx')
