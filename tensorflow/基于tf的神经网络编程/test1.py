import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras

mnist = keras.datasets.mnist.load_data()
# 导入训练集的输入和标签   导入测试集的输入和标签
(train_fea,train_tar),(test_fea,test_tar) = mnist
train_fea,test_fea = tf.layers.flatten(train_fea),tf.layers.flatten(test_fea)
train_tar,test_tar = tf.one_hot(train_tar,10),tf.one_hot(test_tar,10)
val_fea,val_tar = train_fea[-5000:],train_tar[-5000:]
train_fea,train_tar = train_fea[:-5000],train_tar[:-5000]

print(train_fea.shape,train_tar.shape,test_fea.shape,test_tar.shape,val_fea.shape,val_tar.shape)

lr = 0.01
batch_size = 128
epochs = 100

fea = tf.placeholder(tf.float32,[None,784])
tar = tf.placeholder(tf.float32,[None,10])

w1 = tf.Variable(tf.random_normal(shape=[fea.shape[1],10],stddev=0.1))
b1 = tf.Variable(tf.zeros(shape=[10,]))

logits = tf.matmul(fea,w1) + b1
probs = tf.nn.softmax(logits)
cross_entropy = -tf.reduce_sum(tar * tf.log(probs),axis=1) #交叉墒
cost = tf.reduce_mean(cross_entropy) #降到一纬的数据

train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost) #训练
true_list = tf.equal(tf.arg_max(logits,1),tf.arg_max(tar,1))
success_rate = tf.reduce_mean(tf.cast(true_list,tf.float32)) #准确率
init = tf.global_variables_initializer()

# 这段代码是可以保存
def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches
    for ii in range(0, n_batches * batch_size, batch_size):
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y


with tf.Session() as sess:
    sess.run(init)
    total_batch = int(int(train_fea.shape[0])/batch_size)+1
    print(total_batch)
    for epoch in range(epochs):
        for x, y in get_batches(train_fea.eval(), train_tar.eval()):
            sess.run(train_op, feed_dict={fea: x, tar: y})
        if (epoch % 10) == 0:
            val_succ = sess.run(success_rate,feed_dict={fea:val_fea.eval(),tar:val_tar.eval()})
            print('epoch:{0:<5},val_succ:{1:.2f}%'.format(epoch,val_succ*100))
    test_succ = sess.run(success_rate, feed_dict={fea: test_fea.eval(), tar: test_tar.eval()})
    print('epoch:{0:<5},test_succ:{1:.2f}%'.format(epoch, test_succ * 100))










