from random import random
from keras import metrics
from PIL import Image

import numpy as np
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.applications import VGG16
from keras.models import Model
from keras import optimizers
import pandas as pd

data_pos = []
data_negative = []
labels_pos = []
labels_negative = []

angles = [90,180,270]
def rotate_image(image_path, angle,ind,filename=None):
    img = Image.open(image_path)
    rotated_img = img.rotate(angle)
    if(filename!= None):
        rotated_image_path = f"./DATASET_TRAIN/ImgsAllBack/rot/_rotated_{angle}_{filename}"
    else:
        rotated_image_path = f"./DATASET_TRAIN/ImgsAllBack/rot/rotated_{ind}_{angle}.png"
    rotated_img.save(rotated_image_path)

def augment_dataset(folder):
    ind = 0
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(
                ".jpeg"):
            image_path = os.path.join(folder, filename)
            for angle in angles:
                #print(angle)
                rotate_image(image_path, angle,ind,filename)
            ind+=1

def recordToData_Image(folder, data, labels=None, dsize=(100, 100), countImage=None):
    ind = 0
    for filename in os.listdir(folder):
        if countImage != None and ind >= countImage:
            return
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path)
            width, height = image.size
            if labels != None:
                dat_line = f"{image_path} 1 0 0 {width} {height}\n"
                labels.append(dat_line)
            #data.append(cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY), dsize,interpolation=cv2.INTER_AREA))
            data.append(cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dsize,interpolation=cv2.INTER_AREA))
        ind+=1
    return data,labels

def createTrainPos(x_train,y_train):
    for i in range(len(data_pos)):
        x_train[i, :, :, :] = data_pos[i]
        y_train.append(1)
    return x_train,y_train

def createTrainNeg(x_train,y_train):
    total = len(data_pos)  if len(data_pos) < len(data_negative) else len(data_negative)
    curind = len(data_negative)  if len(data_pos) < len(data_negative) else len(data_pos)
    for i in range(total):
        ind = curind + i
        x_train[ind, :, :, :] = data_negative[i]
        y_train.append(2)
    return x_train,y_train

def readyTrainData(folderP,folderN,countPos=None,countNeg=None,sizeW=100,sizeH=100,chanel=1):
    recordToData_Image(folderP, data_pos, labels_pos, countImage=countPos)
    recordToData_Image(folderN, data_negative, labels_negative, countImage=countNeg)
    #print(len(data_pos))
    x_train = np.zeros([len(data_pos)+ len(data_negative) , sizeW, sizeH, chanel])
    y_train = []

    x_train, y_train = createTrainPos(x_train, y_train)
    x_train, y_train = createTrainNeg(x_train, y_train)

    y_train = np.array(pd.get_dummies(y_train))
    return x_train,y_train

def readyTestData(sizeW=100,sizeH=100,chanel=1,countTest=50):
    x_test = np.zeros([countTest, sizeW, sizeH, chanel])
    y_test = []

    for j in range(countTest):
        if j % 2 == 0:
            rnd = np.random.randint(0, len(data_pos)-1)
            x_test[j, :, :, 0] = data_pos[rnd]
            y_test.append(1)
        else:
            rnd = np.random.randint(0, len(data_negative)-1)
            x_test[j, :, :, 0] = data_negative[rnd]  # np.ones((100,100))
            y_test.append(2)

    y_ts = np.array(y_test)
    y_test = np.array(pd.get_dummies(y_test))
    return x_test,y_test


def readyTestData2(sizeW=100, sizeH=100, chanel=1, countTest=50):
    x_test = np.zeros([countTest, sizeW, sizeH, chanel])
    y_test = []
    data_pos1 =[]
    data_negative2=[]
    data_pos1,_ = recordToData_Image('./TestDataset/p/', data_pos1, labels_pos)
    data_negative1,_ =  recordToData_Image('./TestDataset/p_n', data_negative2, labels_negative)

    indPos = 0
    indNeg = 0

    for j in range(countTest):
        if j % 2 == 0:
            if(indPos >= len(data_pos1)):
                indPos = 0
            x_test[j, :, :, :] = data_pos1[indPos]
            y_test.append(1)
            indPos+=1
        else:
            if (indNeg >= len(data_negative1)):
                indNeg = 0
            x_test[j, :, :, :] = data_negative1[indNeg]
            y_test.append(2)
            indNeg +=1

    y_ts = np.array(y_test)
    y_test = np.array(pd.get_dummies(y_test))
    return x_test, y_test

def startTrainingModel(x_train,y_train,x_test=None,y_test=None,epochs=10,channel=3,sizeH=100,sizeW=100):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(sizeW, sizeH, channel))

    # Замораживаем слои базовой модели
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # для уменьшения переобучения модели
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.Precision()])

    # model = Sequential()
    # # первый уровень:
    # model.add(Conv2D(32, (3, 3), input_shape=(sizeW, sizeH, channel), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # # Второй этаж:
    # model.add(Conv2D(128, (3, 3), activation='relu'))  # model.add(Dropout(0.25))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.25))
    # #  2. Полностью связанный слой и выходной слой:
    # model.add(Flatten())
    # # model.add(Dense(500,activation='relu'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))

    # model.summary()
    # model.compile(loss='categorical_crossentropy',  # ,'binary_crossentropy'
    #               optimizer='adam',#optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=1e-06),
    #               metrics=[metrics.Precision()])

    model.fit(x_train, y_train, batch_size=2, epochs=epochs)

    if(len(x_test)!=0):
        y_predict = model.predict(x_test)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f'Точность на тестовых данных: {test_acc}')
        print('Test loss:', test_loss)
    model.save('firstTrainedModel.h5')

def continueTrainingModel(trainedModel,newModal,x_train,y_train,x_test=[],y_test=[],epochs=5):
    model = load_model(trainedModel)

    model.compile(loss='categorical_crossentropy',  # ,'binary_crossentropy'
                  optimizer='adam',
                  metrics=[metrics.Precision()])

    model.fit(x_train, y_train, epochs=epochs, batch_size=2)
    if(len(x_test)!=0):
        y_predict = model.predict(x_test)
        score = model.evaluate(x_test, y_test)
        print(score)
    model.save(newModal)

if __name__ == '__main__':
    x_train,y_train = readyTrainData('./DATASET_TRAIN/DatasetForHaar_p','./n',countNeg=200,chanel=3)
    x_test,y_test = readyTestData2(countTest=37,chanel=3)
    startTrainingModel(x_train,y_train,x_test,y_test,epochs=10)

    #continueTrainingModel('firstTrainedModel.h5','bestTrainedModel.h5',x_train,y_train,x_test=x_test,y_test=y_test,epochs=25)
    #augment_dataset('./DATASET_TRAIN/ImgsAllBack')

