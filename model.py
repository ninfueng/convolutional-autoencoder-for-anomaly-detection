import tensorflow as tf


def unpool(value, name='unpool'):
    """
    From: https://github.com/tensorflow/tensorflow/issues/2169
    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :param name: A string for scope name.
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


class CONVOLUTIONAL_AUTOENCODER(tf.keras.Model):
    def __init__(self, num_neuron=256, kernal1=32, kernal2=16, shape=(32, 32, 3)):
        assert type(num_neuron) == int and type(kernal1) == int and type(kernal2) == int and len(shape) == 3
        super().__init__()
        if not tf.executing_eagerly():
            raise NotImplementedError('Eager execution is needed but it return as : {}'.format(tf.executing_eagerly()))
        else:
            init = tf.contrib.layers.xavier_initializer()
            pooled_shape = (shape[0]/4, shape[1]/4, shape[2])
            self.conv1 = tf.keras.layers.Conv2D(filters=kernal1,
                                                kernel_size=5,
                                                padding='SAME',
                                                activation=tf.nn.relu,
                                                kernel_initializer=init)
            self.max1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                  strides=(2, 2),
                                                  padding='SAME')
            self.conv2 = tf.keras.layers.Conv2D(filters=kernal2,
                                                kernel_size=5,
                                                padding='SAME',
                                                activation=tf.nn.relu,
                                                kernel_initializer=init)
            self.max2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                  strides=(2, 2),
                                                  padding='SAME')
            self.flatten1 = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(units=num_neuron,
                                                activation=tf.nn.relu,
                                                kernel_initializer=init)
            self.dense2 = tf.keras.layers.Dense(units=pooled_shape[0]*pooled_shape[1]*kernal2,
                                                activation=tf.nn.relu,
                                                kernel_initializer=init)
            self.deflatten1 = tf.keras.layers.Reshape(target_shape=(pooled_shape[0], pooled_shape[1], kernal2))
            self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=kernal1, kernel_size=5, padding='SAME',
                                                           activation=tf.nn.relu,
                                                           kernel_initializer=init)
            self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=shape[2], kernel_size=5, padding='SAME',
                                                           activation=tf.nn.sigmoid,
                                                           kernel_initializer=init)

    def call(self, img):
        """
        Pass the batch of images to forward propagate into the networks.
        :param img: Input images with shape as [batch_size, image_size, image_size, image_channel].
        :return: Reconstruction images with shape as same as img.
        """
        x = self.conv1(img)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.deflatten1(x)
        x = unpool(x)
        x = self.deconv1(x)
        x = unpool(x)
        x = self.deconv2(x)
        return x
