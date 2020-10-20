import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from MyFunctions import *


def CNN(Myloss,Myactivation,data):

    parameters_path = "./parameters/"
    data = data
    output_dic = {}

    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    inputs = keras.Input(shape=input_shape)
    h = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    h = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(h)
    h = layers.MaxPooling2D(pool_size=(2, 2))(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(h)
    h = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(h)
    h = layers.MaxPooling2D(pool_size=(2, 2))(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(512, activation="relu")(h)
    h = layers.Dropout(0.5)(h)
    outputs = layers.Dense(num_classes, activation=Myactivation, name="output_layer")(h)


    model = Model(inputs=inputs, outputs=outputs)

    output_layer_features = model.get_layer("output_layer").input
    model2 = Model(inputs=inputs, outputs=output_layer_features)

    model.summary()
    # print(Myactivation)

    batch_size = 128
    epochs = 15

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss=Myloss, optimizer="adam", metrics=["accuracy"])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


    W = model.get_layer("output_layer").kernel
    b = model.get_layer("output_layer").bias
    b = tf.expand_dims(b, 0)
    O = tf.concat([W , b], 0)
    O = tf.transpose(O)

    Feature_train = model2.predict(x_train)
    Feature_test = model2.predict(x_test)

    output_dic["Feat_train"]=tf.transpose(Feature_train)
    output_dic["Feat_test"]=tf.transpose(Feature_test)
    output_dic["T_train"]=tf.transpose(y_train)
    output_dic["T_test"]=tf.transpose(y_test)
    output_dic["O"]=O.numpy() 

    architecture_name = "weights_CNN_"+Myloss+"_"+Myactivation
    save_dic(output_dic, parameters_path, data, architecture_name)



