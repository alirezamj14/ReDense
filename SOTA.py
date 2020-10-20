import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from MyFunctions import *
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.preprocessing.image import ImageDataGenerator


def SOTA():
    
    parameters_path = "./parameters/"
    data = "CIFAR10"
    output_dic = {}

    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)
    batch_size = 128
    epochs = 15

    ##########################################################################################
    ##########################################################################################
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # x_train = preprocess_input(x_train)
    # x_test = preprocess_input(x_test)

    # datagen = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
    # datagen.fit(x_train)

    # # Scale images to the [0, 1] range
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255
    # # Make sure images have shape (28, 28, 1)
    # # x_train = np.expand_dims(x_train, -1)
    # # x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    

    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    base_model.summary()
    # base_model.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    
    # x = keras.layers.UpSampling2D()(inputs)
    # x = keras.layers.UpSampling2D()(x)
    # x = keras.layers.UpSampling2D()(x)

    x = base_model(inputs, training=False)
    # x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output_layer")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    # model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save('my_resnet50_cifar10.h5')

    # base_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # score_base_model = base_model.evaluate(x_test, y_test, verbose=0)
    # print("Test loss:", score_base_model[0])
    # print("Test accuracy:", score_base_model[1])
