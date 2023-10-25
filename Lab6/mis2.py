import cv2
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras import layers, models
from keras.layers import Dense, Flatten
from keras.utils import to_categorical



if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Создание модели
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    '''layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)): Этот код создает сверточный слой (Convolutional Layer). Важные аргументы:
32: Это количество фильтров, которые этот сверточный слой будет использовать для извлечения признаков из входных данных.
(3, 3): Это размер фильтра свертки (3x3), который будет скользить по входным данным для выполнения операции свертки.'''
    model.add(layers.MaxPooling2D((2, 2)))
    '''Эта строка добавляет слой пулинга  к модели. Пулинг используется для уменьшения размерности и извлечения наиболее важных признаков из предыдущего сверточного слоя. (2, 2) указывает на размер окна, которое скользит по данным и выбирает максимальное значение.'''
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu')) #полносвязный слой
    model.add(layers.Dense(10, activation='softmax'))


    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
    '''validation_split=0.2:  20% обучающих данных будут использованы для валидации в каждой эпохе обучения. Это помогает отслеживать производительность модели на данных, которые не использовались в обучении.'''
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Точность на тестовых данных: {test_acc}')
