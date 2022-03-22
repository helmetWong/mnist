import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test_images = x_test
x_train, x_test = x_train / 255.0, x_test / 255.0

loaded_model = "../models/v1_mnist.h5"
model = tf.keras.models.load_model(loaded_model)
model.summary()

#print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
#print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

test_predictions = model.predict(x_test)
predict_labels = np.argmax(test_predictions, axis=1)

index = 0
row = 8
col = 8

plt.figure (figsize = (15,15))
for i in range(0, row * col):
    plt.subplot(row, col, i + 1)
    plt.title('Pre:' + str(predict_labels[index + i]) + 
              ' / Actual:'+ str(y_test[index + i]))
    plt.imshow(x_test[index+ i].reshape(28,28,1), cmap='gray')
    plt.grid(False)
    plt.axis('off')

plt.show()