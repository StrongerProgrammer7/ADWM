from keras.datasets import mnist
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt


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
    model.add(layers.MaxPooling2D((2, 2)))
    '''This line adds a pooling layer to the model. Pooling is used to reduce dimensionality and extract the most important features from the previous convolutional layer. (2, 2) indicates the size of the window that slides over the data and selects the maximum value.'''
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu')) #полносвязный слой
    model.add(layers.Dense(10, activation='softmax'))


    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
    '''validation_split=0.2:  20% of the training data will be used for validation at each training epoch. This helps to track the performance of the model on data that was not used in training.'''
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Точность на тестовых данных: {test_acc}')

    print(test_labels[6])
    plt.imshow(test_images[6], cmap='binary')
    plt.axis('off')
    plt.show()