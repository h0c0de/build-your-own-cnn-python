"""CNN MNIST lengkap (Victor Zhou, CNNs Part 1 & 2).

Conv3x3(8) -> MaxPool2 -> Softmax(1352, 10), dilatih
dengan backprop penuh persis protokol panduan:
1000 citra latih, 3 epoch, lr 0.005, lalu uji 1000 citra.
Jalankan: python3 cnn.py (butuh aset/mnist.npz).
"""

import os

import numpy as np

from konv import Conv3x3
from pool import MaxPool2
from softmax import Softmax

conv = Conv3x3(8)                   # 28x28x1 -> 26x26x8
pool = MaxPool2()                   # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the
    accuracy and cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to
    # make it easier to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy.
    # np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - im is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc


def muat_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "aset", "mnist.npz")
    data = np.load(path)
    return (data["x_train"][:1000].astype(float),
            data["y_train"][:1000],
            data["x_test"][:1000].astype(float),
            data["y_test"][:1000])


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = muat_data()

    print('MNIST CNN initialized!')

    # Train the CNN for 3 epochs
    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))

        # Shuffle the training data
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        # Train!
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i > 0 and i % 100 == 99:
                print('[Step %d] Past 100 steps: Average Loss %.3f | '
                      'Accuracy: %d%%' % (i + 1, loss / 100, num_correct))
                loss = 0
                num_correct = 0

            l, acc = train(im, label)
            loss += l
            num_correct += acc

    # Test the CNN
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_images)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)
