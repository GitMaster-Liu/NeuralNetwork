# coding=utf8
import random
import numpy as np
import tensorflow as tf
import xlwt
from sklearn import svm

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

# 给x，y留出占位符，以便未来填充数据
x = tf.placeholder("float", [None, 60])
y_ = tf.placeholder("float", [None, 8])
# 设置输入层的W和b
W = tf.Variable(tf.zeros([60, 8]))
b = tf.Variable(tf.zeros([8]))
# 计算输出，采用的函数是softmax（输入的时候是one hot编码）
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 第一个卷积层，5x5的卷积核，输出向量是32维
w_conv1 = weight_variable([5, 5, 1, 60])
b_conv1 = bias_variable([60])

x_image = tf.reshape(x, [-1, 10, 6, 1])
print(x_image)
# 图片大小是16*16，,-1代表其他维数自适应
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = avg_pool_2x2(h_conv1)
# 采用的最大池化，因为都是1和0，平均池化没有什么意义

# 第二层卷积层，输入向量是32维，输出64维，还是5x5的卷积核
w_conv2 = weight_variable([5, 5, 60, 60])
b_conv2 = bias_variable([60])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = avg_pool_2x2(h_conv2)

# 全连接层的w和b
w_fc1 = weight_variable([60, 12])
b_fc1 = bias_variable([12])
# 此时输出的维数是256维
h_pool2_flat = tf.reshape(h_pool2, [-1, 60])
h_fc1 = tf.matmul(h_pool2_flat, w_fc1) + b_fc1

#print (type(h_fc1))
# h_fc1是提取出的256维特征，很关键。后面就是用这个输入到SVM中

# 设置dropout，否则很容易过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层，在本实验中只利用它的输出反向训练CNN，至于其具体数值我不关心
w_fc2 = weight_variable([12, 8])
b_fc2 = bias_variable([8])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 设置误差代价以交叉熵的形式
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 用adma的优化算法优化目标函数
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver(max_to_keep=1)
for file_num in range(1,9):
    # 在十个随机生成的不相干数据集上进行测试，将结果综合
    print ('testing NO.%d dataset.......' % file_num)
    ff = open('vector' + file_num.__str__() + '.data')
    rr = ff.readlines()
    x_test2 = []
    y_test2 = []
    for i in range(len(rr)):
        x_test2 = x_test2+(list(map(float,rr[i].split())))
    print(1)
    print(x_test2)
    #print(x_test2)
    x_test2 = np.array(x_test2)
    print(2)
    print(x_test2)
    x_test2 = x_test2.reshape(1, 60)
    print(3)
    print(x_test2)
    #print (x_test2)
    x_test2=x_test2[0]
    print(4)
    print(x_test2)
        #x_test2 = tf.reshape(x_test2, [-1, 256, 256, 1])
        #y_test2.append(list(map(int, rr[i].split(' ')[256:266])))
    ff.close()
    # 以上是读出训练数据
    ff2 = open('position' + file_num.__str__() + '.data')
    rr2 = ff2.readline()

    y_test2 = y_test2+(list(map(float, rr2.split())))
    print(1)
    print(y_test2)
    y_test2 = np.array(y_test2)
    print(2)
    print(y_test2)
    y_test2 = y_test2.reshape(1, 8)
    print(3)
    print(y_test2)
    y_test2=y_test2[0]
    print(4)
    print(y_test2)
    #x_test3 = []
    #y_test3 = []
    #for i in range(len(rr2)):
        #x_test3.append(list(map(int, list(map(float, rr2[i].split(' ')[:256])))))
        #y_test3.append(list(map(int, rr2[i].split(' ')[256:266])))
    #ff2.close()
    # 以上是读出测试数据
    batch = ([], [])
    batch[0].append(x_test2)
    batch[1].append(y_test2)
    for i in range(3000):
        if i < 200:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, train accuracy %g" % (i, train_accuracy))
    # 跑3000轮迭代，每次随机从训练样本中抽出50个进行训练
    #print (type(x_test2))
    #print(type(y_test2))

    #x_temp = x_test2#.eval(session=sess)
    #y_temp = y_test2
    # batch = ([], [])
    # #p = random.sample(range(795), 50)
    # p = range(49)
    #for k in p:
        #batch[0].append(x_test2[k])
        # batch[1].append(y_test2[k])
    #if i % 100 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        #print ("step %d, train accuracy %g" % (i, train_accuracy))
    #print(x_temp)
    #print(y_temp)
        #print(batch)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})
    saver.save(sess, './cnnvector1.ckpt', global_step=file_num)
style = set_style('Times New Roman', 220, True)
f = xlwt.Workbook()
sheet4 = f.add_sheet('Sheet4',cell_overwrite_ok=True)
for file_num in range(1, 9):
    ff = open('vector' + file_num.__str__() + '.data')
    rr = ff.readlines()
    print (rr)
    x_test2 = []
    for i in range(len(rr)):
        x_test2 = x_test2 + (list(map(float, rr[i].split())))
    print(1)
    print(x_test2)
    # print(x_test2)
    x_test2 = np.array(x_test2, dtype='float32')
    print(2)
    print(x_test2)
    #x_test2 = x_test2.reshape(1, 60)
    x_test2 = x_test2.reshape(1, 60)
    print(3)
    print(x_test2)
    # print (x_test2)
    #x_test2 = x_test2[0]
    #x_test2 = np.array(x_test2)
    #x_test2 = x_test2.reshape(1, 60)
    #x_test2=x_test2[0]
    #batch = ([], [])
    #batch[0].append(x_test2)
    #x_1=tf.convert_to_tensor(batch[0])
    #x_1 = tf.reshape(x_1, [-1, 6, 10, 1])

    # 图片大小是16*16，,-1代表其他维数自适应
    #output = tf.nn.relu(conv2d(x_test2, w_conv1) + b_conv1)
    #batch = ([], [])
    #batch[0].append(x_test2)
    #output = sess.run(h_fc1, feed_dict={x: batch[0]})
    output = sess.run(h_fc1, feed_dict={x: x_test2})
    #h_pool1 = max_pool_2x2(output)
    # 此时输出的维数是256维
    #h_pool2_flat = tf.reshape(h_pool1, [-1, 3])
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)



    #print(sess.run(h_fc1, feed_dict={x: batch[0]}))

    print('****')
    #print(batch[0])
    print(output)
    output=output[0]

    print(output)
    vector=output
    vector=vector.tolist()


    #vector=output[0]
    #vector=vector[9]
    #vector=vector[5]


    #outvector = vector*10
    for j in range(len(vector)):
        sheet4.write(file_num-1, j, vector[j], style)
    print('-----------------------------------------------')

f.save(r'C:\Users\lenovo\Desktop\texttezheng.xls')