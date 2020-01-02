import tensorflow as tf
import numpy as np


def load_cifar10(num_batch_train, num_batch_test, mode='RGB'):
    """Load CIFAR-10 dataset with Tensorflow built-in function.
    Generate and shuffle CIFAR-10 iterators via using tf.data.Data.
    :param num_batch_train: An integer.
    :param num_batch_test: An integer.
    :param mode: String in {'RGB', 'HSV', 'TUV'}
    :return train_slice, test_slice: Iterator for training and testing data.
    """
    assert isinstance(num_batch_train, int)
    assert isinstance(num_batch_test, int)
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    # By converting dtype to make Tensorflow automatically use GPU.
    x_train = tf.image.convert_image_dtype(x_train, tf.float32)
    x_test = tf.image.convert_image_dtype(x_test, tf.float32)
    if mode == 'RGB':
        pass
    elif mode == 'HSV':
        x_train = tf.image.rgb_to_hsv(x_train)
        x_test = tf.image.rgb_to_hsv(x_test)
    elif mode == 'TUV':
        x_train = tf.image.rgb_to_hsv(x_train)
        x_test = tf.image.rgb_to_hsv(x_test)
        x_train, x_test = hsv_to_tuv(x_train), hsv_to_tuv(x_test)
    else:
        raise NotImplementedError(
            'Input mode is not valid, could be RGB, HSV, TUV, your input: {}'.format(mode))
    train_slice = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_slice = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_slice = train_slice.shuffle(x_train.shape[0])
    train_slice = train_slice.batch(num_batch_train)
    test_slice = test_slice.batch(num_batch_test)
    train_slice = train_slice.prefetch(buffer_size=num_batch_train)
    test_slice = test_slice.prefetch(buffer_size=num_batch_test)
    return train_slice, test_slice


def inverse_specific_labeled_images(img, label, anomaly_label):
    """Inverse images which has specific labels via negative images and plus 1.
    :param img: Floating point images in range of [0.0, 1.0]
                with shape [batch_size, image_size, image_size, image_channel].
    :param label: A integer tensor with shape [batch_size,].
    :param anomaly_label: An integer in range [0, 9].
    :return img: Images with some batch inversed with same shape as input images.
    """
    assert isinstance(anomaly_label, int)
    assert anomaly_label >= 0
    assert anomaly_label < 10
    mask_anomaly = tf.where(tf.equal(label, anomaly_label), tf.ones(label.shape), tf.zeros(label.shape))
    if len(img._shape_as_list()) == 4:
        batch_size = mask_anomaly.shape[0]
        mask_anomaly = tf.reshape(mask_anomaly, [batch_size, 1, 1, 1])
    elif len(img._shape_as_list()) == 2:
        pass
    else:
        raise NotImplementedError('Your img._shape_as_list(): {}'.format(img._shape_as_list()))
    mask_normal = 1.0 - mask_anomaly
    img = tf.cast(img, tf.float32)
    img = tf.subtract(
        tf.multiply(tf.ones(img.shape), mask_anomaly), tf.multiply(img, mask_anomaly)) + tf.multiply(img, mask_normal)
    return img


def inverse_multiple_labeled_images(img, label, anomaly_label):
    """Inverse images which has specific labels via negative images and plus 1.
    :param img: Floating point images in range of [0.0, 1.0]
                with shape [batch_size, image_size, image_size, image_channel].
    :param label: A integer tensor with shape [batch_size,].
    :param anomaly_label: A list of integer of [0, 9]
    :return img: Images with some batch inversed with same shape as input images.
    """
    assert isinstance(anomaly_label, list)
    mask_anomaly = tf.where(tf.equal(label, anomaly_label[0]), tf.ones(label.shape), tf.zeros(label.shape))
    for idx, i in enumerate(anomaly_label[1::]):
        # Accumulate the anomaly mask from all of elements in anomaly_label.
        # Each of loop will detect each label that in the anomaly_label or not?
        # If True then, put the mask_anomaly as 1 otherwise, 0.
        if idx < len(anomaly_label[1::]):
            # Checking that the idx does not go out of anomaly mask idx.
            mask_anomaly = tf.add(
                tf.where(tf.equal(label, anomaly_label[idx + 1]),
                         tf.ones(label.shape), tf.zeros(label.shape)),
                         mask_anomaly)
    # For checking for given labels and anomaly masks work correctly.
    # print(f'label {label}')
    # print(f'mask_anomaly {mask_anomaly}')

    if len(img._shape_as_list()) == 4:
        batch_size = mask_anomaly.shape[0]
        mask_anomaly = tf.reshape(mask_anomaly, [batch_size, 1, 1, 1])
    elif len(img._shape_as_list()) == 2:
        pass
    else:
        raise NotImplementedError('Your img._shape_as_list(): {}'.format(
            img._shape_as_list()))
    mask_normal = 1.0 - mask_anomaly
    img = tf.cast(img, tf.float32)
    img = tf.subtract(
        tf.multiply(tf.ones(img.shape), mask_anomaly),
        tf.multiply(img, mask_anomaly)) + tf.multiply(img, mask_normal)
    return img


def hsv_to_tuv(hsv_img):
    """Convert HSV space images into TUV space images.
    Note: TUV is purposed image space by Obada Al Alma.
    :param hsv_img: Floating point HSV images in range of [0.0, 1.0]
                    with shape [batch_size, image_size, image_size, image_channel].
                    Degree has an unit as radiance.
    :return: tuv_img: Floating point TUV images with same shape as input images.
    """
    t = tf.sin(hsv_img[:, :, :, 0])*hsv_img[:, :, :, 1]
    u = tf.cos(hsv_img[:, :, :, 0])*hsv_img[:, :, :, 1]
    v = hsv_img[:, :, :, 2]
    tuv_img = tf.stack([t, u, v], axis=-1)
    return tuv_img


def tuv_to_hsv(tuv_img):
    """Converting TUV space images back to HSV space images.
    :param tuv_img: Floating point TUV images in range of [0.0, 1.0]
                    with shape [batch_size, image_size, image_size, image_channel].
                    Degree has an unit as radiance.
    :return hsv_img: Floating point HSV images with same shape as input images.
    """
    h = tf.atan(tuv_img[:, :, :, 0]/tuv_img[:, :, :, 1])
    s = tuv_img[:, :, :, 0]/tf.sin(h)
    v = tuv_img[:, :, :, 2]
    hsv_img = tf.stack([h, s, v], axis=-1)
    return hsv_img


def rearrange_label_loss(locat_label, locat_loss):
    """Load TXT files which contain a label and a loss for each test images.
    Rearrange labels in order and using that orders to reorder losses.
    :param locat_label: A string shows location of TXT file.
    :param locat_loss: A string shows location of TXT file.
    :return rearrange_label, rearrange_loss: Tensors of rearranged labels and losses.
    """
    label = np.loadtxt(locat_label, dtype=np.uint8, delimiter='\n')
    loss = np.loadtxt(locat_loss, dtype=np.float32, delimiter='\n')
    rearrange_label = []
    rearrange_loss = []
    index = np.argsort(label)
    for i in index:
        rearrange_label.append(label[i])
        rearrange_loss.append(loss[i])
    return rearrange_label, rearrange_loss
