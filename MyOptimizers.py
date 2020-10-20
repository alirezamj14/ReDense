# -*- coding: utf-8 -*-

import numpy as np
from MyFunctions import *
import tensorflow as tf
import math

def LS(X_train, T_train, lam):
    """Solve the optimization problem as regularized least-squares"""
    P = X_train.shape[0]
    m = X_train.shape[1]
    
    if P < m:
        Ols = np.dot(np.dot(T_train,X_train.T), np.linalg.inv(np.dot(X_train,X_train.T) + lam * np.eye(P))).astype(np.float32)
    else:
        Ols = np.dot(T_train,np.linalg.inv(np.dot(X_train.T, X_train) + lam * np.eye(m))).dot(X_train.T)
    
    return Ols

def LS_ADMM(Y, T, eps_o, mu, kMax):
    """Optimize O by ADMM method"""
    p=Y.shape[0]
    q=T.shape[0]
    Z, Lam = np.zeros((q, p)), np.zeros((q, p))
    MyTemp = np.linalg.inv(np.dot(Y, Y.T) + 1 / mu * np.eye(p))
    TYT=np.dot(T, Y.T)
    for _ in range(kMax):
        O = np.dot(TYT + 1 / mu * (Z + Lam), MyTemp)
        Z = project_function(O, Lam, eps_o)
        Lam = Lam + Z - O
        
    return O

def project_function(O, Lam, epsilon):
    """Projection for ADMM"""
    Z = O - Lam
    frobenius_norm = math.sqrt(np.sum(Z**2))
    if frobenius_norm > epsilon:
        value = Z * (epsilon/frobenius_norm)
    else:
        value = Z
    
    return value

def Projected_SGD(Yi, T_train, Yi_test, T_test, O_tilde, RB_hparameters):
    
    train_accuracy_lists = []
    train_loss_lists = []
    test_accuracy_lists = []
    test_loss_lists = []
    m = Yi.shape[1]
    P = Yi.shape[0]
    Q= T_train.shape[0]
    data = RB_hparameters["data"]
    learning_rate = RB_hparameters["learning_rate_RB"]
    Epoch_num = RB_hparameters["Epoch_num_RB"]
    batchSize = RB_hparameters["batchSize_RB"]
    optimizer_RB = RB_hparameters["optimizer_RB"]
    eps_o = np.linalg.norm(O_tilde, 'fro')

    print("n = "+str(RB_hparameters["n1_RB"])+", eps_O = "+str(eps_o))

    iteration_num = round(m/batchSize)

    # tf.random.normal(O_tilde.shape)
    # O_init = tf.math.scalar_mul(np.sqrt(1/O_tilde.shape[1]), tf.random.normal(O_tilde.shape))
    # O = tf.Variable(tf.constant(value=O_init), name="O", dtype=tf.float32)
    O = tf.Variable(tf.constant(value=O_tilde), name="O", dtype=tf.float32)
    optimizer = set_optimizer(optimizer_RB, learning_rate)

    @tf.function
    def get_grad(inputs, labels):
        with tf.GradientTape() as tape:
            T_hat = tf.matmul(O, tf.cast(inputs, tf.float32)) 
            total_loss = compute_cost(T_hat, labels)

        gradients = tape.gradient(total_loss, [O])
        return gradients

    train_accuracy_lists = []
    test_accuracy_lists = []
    train_loss_lists=[]
    test_loss_lists=[]
    grad_norm_list=[]

    for epoch in range(1, Epoch_num+1):
        shuffled_Y, shuffled_T = shuffle_data(Yi, T_train)
        
        for i in range(0, iteration_num):
            Y_batch, T_batch = get_batch(shuffled_Y, shuffled_T, i, batchSize)
            gradients = get_grad(Y_batch, T_batch)
            # print("Epoch "+str(epoch)+": "+str(tf.linalg.global_norm(gradients)))
            # print("Maximum gradient: "+str(tf.reduce_max(gradients)))
            optimizer.apply_gradients(zip(gradients, [O]))

            O_norm = np.linalg.norm(O.numpy(), 'fro')
            if O_norm > eps_o:
                O.assign(O.numpy() * eps_o / O_norm)

        T_hat_test = tf.matmul(O, tf.cast(Yi_test, tf.float32)) 
        T_hat = tf.matmul(O, tf.cast(Yi, tf.float32))
        train_accuracy = calculate_accuracy(T_hat.numpy(), T_train)
        test_accuracy = calculate_accuracy(T_hat_test.numpy(), T_test)
        train_loss = compute_cost(T_hat.numpy(), T_train)
        test_loss = compute_cost(T_hat_test.numpy(), T_test)
        train_accuracy_lists.append(train_accuracy)
        test_accuracy_lists.append(test_accuracy)
        train_loss_lists.append(train_loss)
        test_loss_lists.append(test_loss)
        grad_norm_list.append(tf.linalg.global_norm(gradients).numpy())

        if epoch % (10) == 0:
            print("Epoch "+str(epoch)+": Train Accuracy: {:.4f}".format(train_accuracy)," Test Accuracy: {:.4f}".format(test_accuracy))
    print("Epoch "+str(epoch)+": Train Accuracy: {:.4f}".format(train_accuracy)," Test Accuracy: {:.4f}".format(test_accuracy))
    
    Oi = O.numpy()
    return Oi, train_loss_lists, test_loss_lists, train_accuracy_lists, test_accuracy_lists, grad_norm_list

def set_optimizer(Name, learning_rate):
    if Name == "Adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif Name == "Adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif Name == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif Name == "Adamax":
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif Name == "Nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate) 
    elif Name == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)  
    elif Name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)    
    return optimizer


