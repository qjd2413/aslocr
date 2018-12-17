#!/usr/bin/env python
# coding: utf-8

# I used ["Interpret Sign Language with Deep Learning"](https://www.kaggle.com/paultimothymooney/interpret-sign-language-with-deep-learning) code for preprocessing dataset by Paul Mooney, Thanks!

# # Data Preparation

# In[ ]:


import os, cv2, skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Add, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from keras.models import Model,load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir = "./asl-alphabet/asl_alphabet_train/"

# Model saving for easier local iterations
MODEL_DIR = './cnnmodel'
MODEL_PATH = MODEL_DIR + '/cnn-model.h5'
MODEL_WEIGHTS_PATH = MODEL_DIR + '/cnn-model.weights.h5'
MODEL_SAVE_TO_DISK = os.getenv('KAGGLE_WORKING_DIR') != '/kaggle/working'

print('Save model to disk? {}'.format('Yes' if MODEL_SAVE_TO_DISK else 'No'))


def load_model_from_disk():
    '''A convenience method for re-running certain parts of the
    analysis locally without refitting all the data.'''
    model_file = Path(MODEL_PATH)
    model_weights_file = Path(MODEL_WEIGHTS_PATH)

    if model_file.is_file() and model_weights_file.is_file():
        print('Retrieving model from disk...')
        model = load_model(model_file.__str__())

        print('Loading CNN model weights from disk...')
        model.load_weights(model_weights_file)
        return model

    return None


CNN_MODEL = load_model_from_disk()
REPROCESS_MODEL = (CNN_MODEL is None)

print('Need to reprocess? {}'.format(REPROCESS_MODEL))

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                    
                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
#                     X.append(img_arr)
#                     y.append(label)
#     X = np.asarray(X)
#     y = np.asarray(y)
    return X,y

def the_data():
    X_train, y_train = get_data(train_dir)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    y_trainHot = to_categorical(y_train, num_classes=num_classes)
    y_testHot = to_categorical(y_test, num_classes=num_classes)

    X_train.shape, y_trainHot.shape, X_test.shape, y_testHot.shape
        #return X_train, y_trainHot, X_test, y_testHot


    # # Data Augmentation

    # In[ ]:


    train_image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    val_image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
    )

    train_generator = train_image_generator.flow(x=X_train, y=y_trainHot, batch_size=batch_size, shuffle=True)
    val_generator = val_image_generator.flow(x=X_test, y=y_testHot, batch_size=batch_size, shuffle=False)
    return train_generator, val_generator


# # Model

# In[ ]:

def build_model():
    inputs = Input(shape=target_dims)
    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
    net = LeakyReLU()(net)

    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
    net = LeakyReLU()(net)

    shortcut = net

    net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)
    net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)

    net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)
    net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)

    net = Add()([net, shortcut])

    net = GlobalAveragePooling2D()(net)
    net = Dropout(0.2)(net)

    net = Dense(128, activation='relu')(net)
    outputs = Dense(num_classes, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    model.summary()
    return model


# # Training

# In[ ]:

if REPROCESS_MODEL:
    print("in the reprocessing")
    train_generator,val_generator = the_data()
    model = build_model()
    CNN_MODEL = model
    CNN_MODEL.save(MODEL_PATH)
#print("here")
#print(CNN_MODEL.summary())

import datetime

def fit():
    start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    model.fit_generator(train_generator, epochs=5, validation_data=val_generator,
        steps_per_epoch=200,#train_generator.__len__(),
        validation_steps=val_generator.__len__(),
        callbacks=[
            # TensorBoard(log_dir='./logs/%s' % (start_time)),
            # ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto')
    ])
    CNN_MODEL.save_weights(MODEL_WEIGHTS_PATH)

if REPROCESS_MODEL:
    fit()

print("last")
print(CNN_MODEL.summary())

# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on


#
print("Attempting to predict using saved model")
# dimensions of our images
img_width, img_height = 64, 64

# load the model we saved

CNN_MODEL.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#predicting images
img = image.load_img('./asl-alphabet/asl_alphabet_test/G_test.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = CNN_MODEL.predict(images, batch_size=10)
print(classes)

# predicting multiple images at once
# img = image.load_img('./asl-alphabet/asl_alphabet_test/K_test.jpg', target_size=(img_width, img_height))
# y = image.img_to_array(img)
# y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
# images = np.vstack([x, y])
# classes = CNN_MODEL.predict(images, batch_size=10)
# predictions  = []
# for filename in os.listdir('./asl-alphabet/asl_alphabet_test/'):
#     name = './asl-alphabet/asl_alphabet_test/' + filename
#     img = image.load_img(name, target_size=(img_width, img_height))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#     classes = CNN_MODEL.predict(images, batch_size=10)
#     predictions.append(np.argmax(classes, axis=1)[0])
#
# print(predictions)
# print the classes, the images belong to
#print(classes)
# print("argmax")
# print(np.argmax(classes,axis=1))
#print (classes[0])
#print (classes[0][0])

# In[ ]:




