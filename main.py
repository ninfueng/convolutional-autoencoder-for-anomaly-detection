import tensorflow as tf
from model import CONVOLUTIONAL_AUTOENCODER
from dataset import load_cifar10, inverse_multiple_labeled_images
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np
# from matplotlib.colors import hsv_to_rgb

parser = argparse.ArgumentParser(description='Tensorflow implementation of autoencoder for anomaly detection')
parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of training epoch')
parser.add_argument('--learning_rate', '-l', type=float, default=9.0, help='Learning rate')
parser.add_argument('--train_batch', '-trb', type=int, default=128, help='Train batch amount')
parser.add_argument('--test_batch', '-tb', type=int, default=10000, help='Test batch amount')
parser.add_argument('--num_neuron', '-nn', type=int, default=256,
                    help='Number of neurons in fully connected layer for produce codes')
parser.add_argument('--step_down', '-sd', type=int, default=40, help='Step down epoch')
parser.add_argument('--anomaly_label', '-al', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='Anomaly label')
parser.add_argument('--normal_label', '-nl', type=list, default=[0], help='List of ')
parser.add_argument('--mode', '-m', type=str, default='RGB', help='Load cifar10 in selected format (RGB, HSV or TUV)')
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    tf.enable_eager_execution()
    autoencoder = CONVOLUTIONAL_AUTOENCODER(num_neuron=args.num_neuron, kernal1=32, kernal2=16, shape=(32, 32, 3))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    train_data, test_data = load_cifar10(
                            num_batch_train=args.train_batch,
                            num_batch_test=args.test_batch,
                            mode=args.mode)
    t1 = time.time()
    for i in range(args.epoch):
        if i != 0 and i % args.step_down == 0:
            args.learning_rate /= 2

        accumulate_train_loss = []
        for train_img, train_label in train_data.make_one_shot_iterator():
            with tf.GradientTape() as tape:
                logits = autoencoder.call(train_img)
                inv_train_img = inverse_multiple_labeled_images(
                                train_img,
                                train_label,
                                args.anomaly_label,
                                args.normal_label)
                loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=inv_train_img, predictions=logits))
                grads = tape.gradient(loss, autoencoder.variables)
                optimizer.apply_gradients(
                    zip(grads, autoencoder.variables),
                    global_step=tf.train.get_or_create_global_step())
                accumulate_train_loss.append(loss)

        print('Epoch: {}'.format(i+1))
        print('Training MSE Loss: {}'.format(tf.reduce_mean(accumulate_train_loss).numpy()))
        print('Learning rate: {}'.format(args.learning_rate))
        print('Timer: {}'.format(time.strftime("%H:%M:%S", time.gmtime(round(time.time() - t1, 2)))))

        accumulate_test_loss = []
        if i == args.epoch-1:
            for test_img, test_label in test_data.make_one_shot_iterator():
                logits = autoencoder.call(test_img)
                inv_test_img = inverse_multiple_labeled_images(
                               test_img,
                               test_label,
                               args.anomaly_label,
                               args.normal_label)
                test_loss = tf.losses.mean_squared_error(labels=inv_test_img, predictions=logits, reduction="none")
                test_loss = tf.reduce_mean(test_loss, axis=[1, 2, 3])
                loss = tf.reduce_mean(test_loss)
                accumulate_test_loss.append(loss)
            print('Testing MSE Loss: {}'.format(tf.reduce_mean(accumulate_test_loss).numpy()))

    num_img_show = 20
    for i in range(num_img_show):
        reshape_logits = logits.numpy()
        plt.subplot(4, 5, i + 1)
        plt.imshow(reshape_logits[i, :, :, :])
    plt.show()
    for i in range(num_img_show):
        plt.subplot(4, 5, i + 1)
        plt.imshow(test_img[i, :, :, :])
    plt.show()
    np.savetxt('test_label.txt', test_label.numpy().astype(int),  fmt='%i')
    np.savetxt('test_loss.txt', test_loss.numpy(), fmt='%f')
