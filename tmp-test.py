import tensorflow as tf
from dataset import load_cifar10, inverse_specific_labeled_images
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
tf.enable_eager_execution()
import math

def inverse_multiple_labeled_images(img, label, anomaly_label, normal_label):
    assert type(anomaly_label) == list and type(normal_label) == list
    mask_anomaly = tf.where(tf.equal(label, anomaly_label[0]), tf.ones(label.shape), tf.zeros(label.shape))
    for i in anomaly_label[1::]:
        mask_anomaly = tf.add(tf.where(tf.equal(label, anomaly_label[i-1]), tf.ones(label.shape),
                                       tf.zeros(label.shape)), mask_anomaly)
    if len(img._shape_as_list()) == 4:
        batch_size = mask_anomaly.shape[0]
        mask_anomaly = tf.reshape(mask_anomaly, [batch_size, 1, 1, 1])
    elif len(img._shape_as_list()) == 2:
        pass
    else:
        raise NotImplementedError('Your img._shape_as_list(): {}'.format(img._shape_as_list()))
    mask_normal = 1.0 - mask_anomaly
    img = tf.cast(img, tf.float32)
    img = tf.subtract(tf.multiply(tf.ones(img.shape), mask_anomaly), tf.multiply(img, mask_anomaly)) \
          + tf.multiply(img, mask_normal)
    return img

def hsv_to_tuv(hsv_img):
    t = tf.sin(hsv_img[:, :, :, 0]) * hsv_img[:, :, :, 1]
    u = tf.cos(hsv_img[:, :, :, 0]) * hsv_img[:, :, :, 1]
    v = hsv_img[:, :, :, 2]
    tuv_img = tf.stack([t, u, v], axis=-1)
    return tuv_img

def tuv_to_hsv(tuv_img):
    h = tf.atan(tuv_img[:, :, :, 0]/tuv_img[:, :, :, 1])
    s = tuv_img[:, :, :, 0]/tf.sin(h)
    v = tuv_img[:, :, :, 2]
    hsv_img = tf.stack([h, s, v], axis=-1)
    return hsv_img


def load_cifar10(num_batch_train, num_batch_test, mode='RGB'):
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = tf.image.convert_image_dtype(x_train, tf.float32)
    x_test = tf.image.convert_image_dtype(x_test, tf.float32)
    print(x_train)

    if mode == 'RGB':
        # x_train, x_test = x_train / 255.0, x_test / 255.0
        pass
    elif mode == 'HSV':
        x_train = tf.image.rgb_to_hsv(x_train)
        x_test = tf.image.rgb_to_hsv(x_test)
    elif mode == 'TUV':
        x_train = tf.image.rgb_to_hsv(x_train)
        x_test = tf.image.rgb_to_hsv(x_test)
        x_train, x_test = hsv_to_tuv(x_train), hsv_to_tuv(x_test)
    else:
        raise NotImplementedError('Input mode is not valid, your input: {}'.format(mode))
    train_slice = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_slice = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_slice = train_slice.shuffle(x_train.shape[0])
    train_slice = train_slice.batch(num_batch_train)
    test_slice = test_slice.batch(num_batch_test)
    train_slice = train_slice.prefetch(buffer_size=num_batch_train)
    test_slice = test_slice.prefetch(buffer_size=num_batch_test)
    return train_slice, test_slice



# def inverse_specific_labeled_images(img, label, anomaly_label):
#     assert type(anomaly_label) == int and anomaly_label >= 0 and anomaly_label < 10
#
#     mask_anomaly = tf.where(tf.equal(label, anomaly_label), tf.ones(label.shape), tf.zeros(label.shape))
#     if len(img._shape_as_list()) == 4:
#         batch_size = mask_anomaly.shape[0]
#         mask_anomaly = tf.reshape(mask_anomaly, [batch_size, 1, 1, 1])
#     elif len(img._shape_as_list()) == 2:
#         pass
#     else:
#         raise NotImplementedError('Your img._shape_as_list(): {}'.format(img._shape_as_list()))
#
#     mask_normal = 1.0 - mask_anomaly
#     img = tf.cast(img, tf.float32)
#     img = tf.subtract(tf.multiply(tf.ones(img.shape), mask_anomaly), tf.multiply(img, mask_anomaly)) + tf.multiply(
#         img, mask_normal)
#     return img

if __name__ == '__main__':

    train_data, test_data = load_cifar10(num_batch_train=128, num_batch_test=10000, mode='HSV')

    for train_img, train_label in train_data.make_one_shot_iterator():
        # out = inverse_multiple_labeled_images(train_img, train_label, [1, 2, 3, 4, 5, 6, 7, 8, 9], [0])
        pass

    train_img_rgb = hsv_to_rgb(train_img)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(train_img_rgb[i, :, :, :])
    plt.show()

    #print(out)
    # print(train_img[0, :, :, :][:, :, 0])
    # print(train_img[0, :, :, :][:, :, 1])
    # print(train_img[0, :, :, :][:, :, 2])
    #
    # print(tf.reduce_max(train_img[0, :, :, :][:, :, 0]))
    # print(tf.reduce_max(train_img[0, :, :, :][:, :, 1]))
    # print(tf.reduce_max(train_img[0, :, :, :][:, :, 2]))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(train_img[i, :, :, :])
    plt.show()


    train_img = hsv_to_tuv(train_img)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(train_img[i, :, :, :])
    plt.show()

    train_img = tuv_to_hsv(train_img)
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(train_img[i, :, :, :])
    plt.show()

    train_img = hsv_to_rgb(train_img)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(train_img[i, :, :, :])
    plt.show()







