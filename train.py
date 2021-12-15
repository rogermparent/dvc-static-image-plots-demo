
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import utils
from collections import defaultdict
from dvclive.keras import DvcLiveCallback

results = defaultdict(list)

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        import time
        logs = logs or {}
        for metric, value in logs.items():
            results[metric].append(value)
        results['epoch'].append(epoch)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    classes = 10
    y_train = utils.to_categorical(y_train, classes)
    y_test = utils.to_categorical(y_test, classes)
    return (x_train, y_train), (x_test, y_test)


def get_model():
    model = Sequential()

    model.add(Dense(256, input_dim=784))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, input_dim=256))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
    metrics=['accuracy'], optimizer='sgd')
    return model


(x_train, y_train), (x_test, y_test) = load_data()
model = get_model()

model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          batch_size=128,
          epochs=5,
          callbacks=[MetricsCallback(), DvcLiveCallback()])

def write_results(results, d):
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    os.makedirs(d)
    x = results.pop('epoch')
    for key, y in results.items():
        plt.figure()
        plt.plot(x, y)
        plt.ylabel(key)
        plt.savefig(os.path.join(d, key + '.png'))


write_results(results, 'plots')

