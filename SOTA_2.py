import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from MyFunctions import *


def SOTA_2():
    
    parameters_path = "./parameters/"
    data = "CIFAR10"
    output_dic = {}

    # Model / data parameters
    num_classes = 100
    input_shape = (32, 32, 3)
    batch_size = 128
    epochs = 15

    #####################################################################################

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet' )
    # model.summary()
    # # create_directory("./DenseNet169") 
    # model.save('./parameters')

    # model = tf.keras.models.load_model('./parameters/wideResNet_28_10_x4107')
    # model = tf.keras.models.load_model('./parameters/ResNet50_cifar10')
    model = tf.keras.models.load_model('./parameters/BiT_cifar10')
    model.summary()
    # print(model.predict(x_test).shape)
    # model = tf.keras.applications.DenseNet201(include_top=True, weights='./parameters/checkpoint.cifar10_densenet.h5')
    # model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # score = model.evaluate(x_train, y_train, verbose=0)
    # print("Train loss:", score[0])
    # print("Train accuracy:", score[1])

    score = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])