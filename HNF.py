# -*- coding: utf-8 -*-

import numpy as np
import logging
from MyFunctions import *
from MyOptimizers import *
import json
import os

def HNF(_logger, X_train, X_test, T_train, T_test, HNF_hparameters):
    """Build High-dimensional Neural Feature(HNF)"""
    # define variables
    Pi= X_train.shape[0]
    Q = T_train.shape[0]
    n1 = HNF_hparameters["n1"]
    mu = HNF_hparameters["mu"]
    lam = HNF_hparameters["lam"]
    kMax = HNF_hparameters["kMax"]
    data = HNF_hparameters["data"]
    LayerNum = HNF_hparameters["LayerNum"]
    
    # initialize collections
    outputs = {}
    train_accuracy_lists = []
    test_accuracy_lists = []
    
    # create a necessary directory
    parameters_path = "./parameters/"
    create_directory(parameters_path)

    N_train = X_train.shape[1]
    N_test = X_test.shape[1]
    X_train=np.concatenate((X_train, np.ones((1,N_train))), axis=0)
    X_test=np.concatenate((X_test, np.ones((1,N_test))), axis=0)
    Yi=X_train
    Yi_test=X_test
    Pi= Yi.shape[0]

    for layer in range(1, LayerNum + 1):
        _logger.info("Begin to optimize layer {}".format(layer))

        Ri = 2 * np.random.rand(n1, Pi) - 1 if layer == 1 else 2 * np.random.rand(Pi, Pi) - 1
        Vi=np.concatenate([np.eye(Ri.shape[0]), (-1) * np.eye(Ri.shape[0])], axis=0)
        Ui=np.concatenate([np.eye(Ri.shape[0]), (-1) * np.eye(Ri.shape[0])], axis=1)
        Wi=np.dot(Vi, Ri)

        Zi = np.dot(Wi, Yi)
        Zi_test = np.dot(Wi, Yi_test)

        Yi = activation(Zi)
        Yi_test = activation(Zi_test)

        Yi=np.concatenate((Yi, np.ones((1,N_train))), axis=0)
        Yi_test=np.concatenate((Yi_test, np.ones((1,N_test))), axis=0)
        
        if layer == 1:
            Oi = LS( X_train, T_train, lam)

            T_hati = np.dot(Oi, X_train) 
            T_hati_test = np.dot(Oi, X_test)

            train_accuracy = calculate_accuracy(T_hati, T_train)
            test_accuracy = calculate_accuracy(T_hati_test, T_test)
            train_accuracy_lists.append(train_accuracy)
            test_accuracy_lists.append(test_accuracy)
            _logger.info("Train accuracy: {:.2f}".format(train_accuracy))
            _logger.info("Test accuracy: {:.2f}".format(test_accuracy))

        Ri_pinv = np.dot(np.linalg.inv(np.dot(Ri.T,Ri)), Ri.T)
        eps_o = np.linalg.norm(np.dot(np.dot(Oi,Ri_pinv),Ui), 'fro')
        
        Oi = LS_ADMM( Yi, T_train, eps_o, mu, kMax)

        T_hati = np.dot(Oi, Yi) 
        T_hati_test = np.dot(Oi, Yi_test)

        train_accuracy = calculate_accuracy(T_hati, T_train)
        test_accuracy = calculate_accuracy(T_hati_test, T_test)
        train_accuracy_lists.append(train_accuracy)
        test_accuracy_lists.append(test_accuracy)
        train_accuracy_listsP = [ '%.2f' % elem for elem in train_accuracy_lists ]
        test_accuracy_listsP = [ '%.2f' % elem for elem in test_accuracy_lists ]
        _logger.info("Train accuracy: {}".format(train_accuracy_listsP))
        _logger.info("Test accuracy: {}".format(test_accuracy_listsP))

        Pi = Yi.shape[0]
        
        # preserve optimized parameters
        outputs["W" +str(layer)] = Wi.astype(np.float32)
        outputs["O"] = Oi.astype(np.float32)                   #   Oi of the last layer will be saved only
        
    _logger.info("Finish constructing neural network")
    
    _logger.info("Saved optimized parameters for backpropagation")
    save_dic(outputs, parameters_path, data, "weights")