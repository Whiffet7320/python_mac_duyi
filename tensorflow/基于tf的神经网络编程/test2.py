import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras as keras
import os
from myMethods import get_batches,getConfig_photo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告

mnist = keras.datasets.mnist.load_data()
# 导入训练集的输入和标签   导入测试集的输入和标签
(train_fea,train_tar),(test_fea,test_tar) = mnist
train_tar,test_tar = tf.one_hot(train_tar,10),tf.one_hot(test_tar,10)
val_fea,val_tar = train_fea[-5000:],train_tar[-5000:]
train_fea,train_tar = train_fea[:-5000],train_tar[:-5000]
train_fea,val_fea,test_fea = tf.expand_dims(train_fea,-1),tf.expand_dims(val_fea,-1),tf.expand_dims(test_fea,-1)
print(train_fea.shape,train_tar.shape,test_fea.shape,test_tar.shape,val_fea.shape,val_tar.shape)

epochs = 10
batch_size = 128
lr = 0.01

input_ = tf.placeholder(tf.float32,[None,28,28,1])
label_ = tf.placeholder(tf.float32,[None,10])

w_conv1 = tf.Variable(tf.random_normal([5,5,1,6],stddev=0.1))
w_conv2 = tf.Variable(tf.random_normal([5,5,6,16],stddev=0.1))
w_fc1 = tf.Variable(tf.random_normal([5*5*16,120],stddev=0.1))
w_fc2 = tf.Variable(tf.random_normal([120,84],stddev=0.1))
w_fc3 = tf.Variable(tf.random_normal([84,10],stddev=0.1))

conv1 = tf.nn.conv2d(input_,w_conv1,[1,1,1,1],padding='SAME')
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME')
conv2 = tf.nn.conv2d(pool1,w_conv2,[1,1,1,1],padding='VALID')
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding='SAME')
flatten1 = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
fc1 = tf.matmul(flatten1,w_fc1)
fc1 = tf.nn.relu(fc1)
fc2 = tf.matmul(fc1,w_fc2)
fc2 = tf.nn.relu(fc2)
logits = tf.matmul(fc2,w_fc3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_,logits=logits))
# op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
op = tf.train.AdamOptimizer().minimize(cost)
true_list = tf.equal(tf.arg_max(logits,1),tf.arg_max(label_,1))
acc_rate = tf.reduce_mean(tf.cast(true_list,tf.float32))


with tf.Session(config=getConfig_photo()) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for x, y in get_batches(train_fea.eval(), train_tar.eval()):
            sess.run(op,feed_dict={input_:x,label_:y})
        val_acc = sess.run(acc_rate,feed_dict={input_:test_fea.eval(),label_:test_tar.eval()})
        print('epoch:{0:<3},val_acc:{1:.2f}%'.format(epoch,val_acc*100))









