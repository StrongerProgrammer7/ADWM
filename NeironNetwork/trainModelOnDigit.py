from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1)) #number of pictures, XxY, Channel number 1 - gray
    train_images = train_images.astype('float32') / 255 #normalize the data by dividing all values by 255. for diapasons 0-1

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = Sequential()

    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(10, activation='softmax'))#exit

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=64)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)


    print(test_labels[6])
    plt.imshow(test_images[6], cmap='binary')
    plt.axis('off')
    plt.show()

''' sequential learning
    for i in range(len(train_images)):
        image = train_images[i:i + 1]  # Выбираем одно изображение
        label = train_labels[i:i + 1]  # Выбираем соответствующую метку

        # Training a model on a single sample
        model.train_on_batch(image, label)'''

'''
Optimizer
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
optimizer = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
optimizer = keras.optimizers.Adagrad(learning_rate=0.01)

'''

'''
loss='categorical_crossentropy': This is a loss function , used to measure the error between model predictions and correct responses during training. 'categorical_crossentropy' is typically used in multi-class categorization, where multiple classes can be assigned to each object.
'''

'''
Precision (Точность): Shows what proportion of positive predictions are actually true.
Recall (Полнота): Indicates what proportion of positive examples the model detects.
'''