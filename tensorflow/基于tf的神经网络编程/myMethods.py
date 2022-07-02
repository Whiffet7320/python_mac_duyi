import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

def getConfig_photo():
    gpu_device_name = tf.test.gpu_device_name()
    print(gpu_device_name)
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    config_photo = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return config_photo


def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches
    for ii in range(0, n_batches * batch_size, batch_size):
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y


