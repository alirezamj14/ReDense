import numpy as np
import tensorflow as tf
from MyFunctions import *
import json
import pickle

def MLP(_logger, X_train, X_test, T_train, T_test, MLP_hparameters):
    """Back propagation based on the architecture HNF has constructed"""
    train_accuracy_lists = []
    train_NME_lists = []
    test_accuracy_lists = []
    test_NME_lists = []
    m = X_train.shape[1]
    P = X_train.shape[0]
    Q= T_train.shape[0]
    data = MLP_hparameters["data"]
    learning_rate = MLP_hparameters["learning_rate"]
    Epoch_num = MLP_hparameters["Epoch_num"]
    batchSize = MLP_hparameters["batchSize"]
    Layer_Num = MLP_hparameters["LayerNum"]

    iteration_num = round(m/batchSize)

    _logger.info("Read parameters by HNF")
    parameters_path = "./parameters/"
    
    outputs = load_dic(parameters_path, data, "weights")

    ######################################################################################
    ####################        Tensorflow v2       ######################################
    ######################################################################################
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            for i in range(1, Layer_Num + 1):
                ni=outputs["W"+str(i)].shape[0]
                mi=outputs["W"+str(i)].shape[1]
                # setattr(self, "W"+str(i), tf.Variable(initial_value=tf.random.normal(outputs["W"+str(i)].shape), name="W"+str(i), dtype=tf.float32))
                setattr(self, "W"+str(i), tf.Variable(tf.constant(value=outputs["W"+str(i)]), name="W"+str(i), dtype=tf.float32))
                self.O = tf.Variable(tf.constant(value=outputs["O"]), name="O", dtype=tf.float32)

        def call(self, inputs):
            Y=inputs
            Y=tf.concat([Y, tf.ones([1, Y.shape[1]], tf.float32)], 0)
            for i in range(1, Layer_Num + 1):
                Z = tf.matmul(getattr(self, "W"+str(i)), Y)
                Y = tf.keras.activations.relu(Z)
                Y=tf.concat([Y, tf.ones([1, Y.shape[1]], tf.float32)], 0)
            T_hat=tf.matmul(self.O, tf.cast(Y, tf.float32)) 
            return T_hat, Y
        
    my_model = MyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def get_grad_and_loss(inputs, labels):
        with tf.GradientTape() as tape:
            T_hat, _=my_model(inputs)
            total_loss = compute_cost(T_hat, labels)

        gradients = tape.gradient(total_loss, my_model.trainable_variables)
        return gradients

    train_accuracy_lists = []
    test_accuracy_lists = []

    for epoch in range(1, Epoch_num+1):
        shuffled_X, shuffled_T = shuffle_data(X_train, T_train)

        for i in range(0, iteration_num):
            X_batch, T_batch = get_batch(shuffled_X, shuffled_T, i, batchSize)
            gradients = get_grad_and_loss(X_batch, T_batch)
            optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))

        if epoch % (Epoch_num/10) == 0:    
            T_hat, _=my_model(X_train)
            T_hat_test, _=my_model(X_test)
            train_accuracy = calculate_accuracy(T_hat.numpy(), T_train)
            test_accuracy = calculate_accuracy(T_hat_test.numpy(), T_test)
            train_accuracy_lists.append(train_accuracy)
            test_accuracy_lists.append(test_accuracy)
            print("Train Accuracy: {:.3f}".format(train_accuracy)," Test Accuracy: {:.3f}".format(test_accuracy))

    _, Y_train = my_model(X_train)
    _, Y_test = my_model(X_test)
    outputs["Feat_train"]=Y_train[:-1,:]
    outputs["Feat_test"]=Y_test[:-1,:]
    outputs["T_train"]=T_train
    outputs["T_test"]=T_test
    outputs["O"]=my_model.O.numpy()

    save_dic(outputs, parameters_path, data, "weights_MLP")


        
