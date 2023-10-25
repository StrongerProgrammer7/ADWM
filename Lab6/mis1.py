import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical



if __name__ == '__main__':
    # Загрузка данных MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Преобразование данных
    train_images = train_images.reshape((60000, 28, 28, 1)) #кол картинок, XxY, Кол каналов 1 - серый
    train_images = train_images.astype('float32') / 255 #нормализацию данных, разделяя все значения на 255. для диап 0-1

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    #преобразовать метки классов в one-hot кодировку.
    #One-hot кодировка - это способ представления категориальных данных, таких как метки классов, в виде бинарных векторов
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = Sequential() #последовательный стек слоев нейронов.

    model.add(Flatten(input_shape=(28, 28, 1))) #добавляет первый слой к модели.
    '''Слой Flatten предназначен для преобразования входных данных в одномерный вектор.'''
    model.add(Dense(128, activation='relu'))
    '''Это полносвязный (Dense) слой с 128 нейронами.Функция ReLU широко используется в нейронных сетях для введения нелинейности.'''
    model.add(Dense(64, activation='relu'))

    model.add(Dense(10, activation='softmax'))#exit
    '''Он также полносвязный, но имеет 10 нейронов, соответствующих 10 классам (цифры от 0 до 9) Слой softmax используется для многоклассовой классификации и выводит вероятности принадлежности объекта к каждому из классов.'''

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #Скомпилируйте модель, указав функцию потерь и оптимизатор.
    model.fit(train_images, train_labels, epochs=5, batch_size=64) #Обучите модель на обучающих данных

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
''' Последовательное обучение
    for i in range(len(train_images)):
        image = train_images[i:i + 1]  # Выбираем одно изображение
        label = train_labels[i:i + 1]  # Выбираем соответствующую метку

        # Обучаем модель на одном образце
        model.train_on_batch(image, label)'''

'''
Optimizer
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
optimizer = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
optimizer = keras.optimizers.Adagrad(learning_rate=0.01)

'''

'''
loss='categorical_crossentropy': Это функция потерь ,  используется для измерения ошибки между предсказаниями модели и верными ответами во время обучения. 'categorical_crossentropy' обычно используется в многоклассовой классификации, где каждому объекту можно присвоить несколько классов. 
'''

'''
Precision (Точность): Показывает, какая доля положительных предсказаний действительно верна. 
Recall (Полнота): Показывает, какую долю положительных примеров модель обнаруживает. 
'''