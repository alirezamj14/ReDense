import numpy as np
import tensorflow as tf
from MyFunctions import *
from MyOptimizers import *
# from scipy.special import softmax
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

def ReBoost(RB_hparameters):
    """Back propagation based on the architecture HNF-softmax has constructed"""
    # create a necessary directory
    n1 = RB_hparameters["n1_RB"]
    data = RB_hparameters["data"]
    LayerNum = RB_hparameters["LayerNum_RB"]
    learning_rate = RB_hparameters["learning_rate_RB"]
    Epoch_num = RB_hparameters["Epoch_num_RB"]
    optimizer_RB = RB_hparameters["optimizer_RB"]
    batchSize = RB_hparameters["batchSize_RB"]
    random_init = RB_hparameters["initialization"]
    architecture_name = RB_hparameters["architecture_name"]

    parameters_path = "./parameters/"
    create_directory(parameters_path)


    result_path = "./results/"
    data_path = result_path + data + "_"
    create_directory(result_path)

    # model_name = "MLP"
    model_name = architecture_name      #"CNN"
    # model_name = "BiT"
    # model_name = "BiT_mixup"
    # model_name = "BiT-M-R101x3"
    # model_name = "BiT-M-R101x3_few"

    outputs = load_dic(parameters_path, data, "weights_"+model_name)

    X_train=outputs["Feat_train"]
    T_train=outputs["T_train"]
    X_test=outputs["Feat_test"]
    T_test=outputs["T_test"]

    p = 0.5
    # X_train = tf.nn.dropout(X_train, rate = 1-p).numpy()
    # X_test = tf.nn.dropout(X_test, rate = 0.5, seed=2).numpy()

    N_train = X_train.shape[1]
    N_test = X_test.shape[1]

    print(X_train.shape[0])
    X_train=np.concatenate((X_train, np.ones((1,N_train))), axis=0)
    X_test=np.concatenate((X_test, np.ones((1,N_test))), axis=0)

    RB_hparameters["batchSize_RB"] = N_train

    Pi= X_train.shape[0]
    Q = T_train.shape[0]

    # initialize collections
    train_accuracy_lists = []
    test_accuracy_lists = []
    
    Yi=X_train;
    Yi_test=X_test;

    for layer in range(1, LayerNum + 1):
        # print("Begin to optimize layer {}".format(layer))
        
        if model_name == "MLP":
            Ri = 2 * np.random.rand(n1+1, Pi) - 1 if layer == 1 else 2 * np.random.rand(n1+1, Pi) - 1
        else:
            Ri = 2 * np.random.rand(Pi, Pi) - 1 if layer == 1 else 2 * np.random.rand(Pi, Pi) - 1

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
            Oi = outputs["O"]

            # T_hati = np.dot(Oi, X_train) 
            T_hati = np.exp(np.dot(Oi, X_train))/np.sum(np.exp(np.dot(Oi, X_train)),axis=0) 
            # T_hati_test = np.dot(Oi, X_test)
            # print(T_hati[:,0])
            T_hati_test = np.exp(np.dot(Oi*p, X_test))/np.sum(np.exp(np.dot(Oi*p, X_test)),axis=0) 

            train_accuracy = calculate_accuracy(T_hati, T_train)
            test_accuracy = calculate_accuracy(T_hati_test, T_test)
            train_accuracy_lists.append(train_accuracy)
            test_accuracy_lists.append(test_accuracy)
            print("Train accuracy: {:.4f}".format(train_accuracy))
            print("Test accuracy: {:.4f}".format(test_accuracy))

        Ri_pinv = np.dot(np.linalg.inv(np.dot(Ri.T,Ri)), Ri.T)
        O_tilde=np.dot(np.dot(Oi,Ri_pinv),Ui).astype(np.float32)
        O_tilde = np.concatenate((O_tilde, np.zeros((Q,1), dtype=np.float32)), axis=1)
        
        Oi, train_loss_GD, test_loss_GD, train_acc_GD, test_acc_GD, grad_norm_GD = Projected_SGD(Yi, T_train, Yi_test, T_test, O_tilde, RB_hparameters)

        plt.subplots()
        plt.plot(range(0, len(test_loss_GD)), np.array(test_loss_GD), 'r-', label="Test Loss")
        plt.plot(range(0, len(train_loss_GD)), np.array(train_loss_GD), 'b-', label="Train Loss")
        plt.legend(loc='best')
        plt.savefig(data_path +"Loss_vs_epoch_"+optimizer_RB+"_neurons_" \
            +str(Yi.shape[0])+"_learningRate_"+str(learning_rate)+"_"+model_name+".png")
        plt.close()

        plt.subplots()
        plt.plot(range(0, len(test_acc_GD)), np.array(test_acc_GD), 'r-', label="Test Accuracy")
        plt.plot(range(0, len(train_acc_GD)), np.array(train_acc_GD), 'b-', label="Train Accuracy")
        plt.legend(loc='best')
        plt.savefig(data_path +"Acc_vs_epoch_"+optimizer_RB+"_neurons_" \
            +str(Yi.shape[0])+"_learningRate_"+str(learning_rate)+"_"+model_name+".png")
        plt.close()

        # print("Norm Oi: {:.2f}".format(np.linalg.norm(Oi, 'fro')))

        # T_hati = softmax(np.dot(Oi, Yi) , axis=0) 
        T_hati = np.exp(np.dot(Oi, Yi))/np.sum(np.exp(np.dot(Oi, Yi)),axis=0) 
        # T_hati_test = softmax(np.dot(Oi, Yi_test), axis=0)
        T_hati_test = np.exp(np.dot(Oi*p, Yi_test))/np.sum(np.exp(np.dot(Oi*p, Yi_test)),axis=0) 

        train_accuracy = calculate_accuracy(T_hati, T_train)
        test_accuracy = calculate_accuracy(T_hati_test, T_test)
        train_accuracy_lists.append(train_accuracy)
        test_accuracy_lists.append(test_accuracy)
        train_accuracy_listsP = [ '%.4f' % elem for elem in train_accuracy_lists ]
        test_accuracy_listsP = [ '%.4f' % elem for elem in test_accuracy_lists ]
        print("Train accuracy: {}".format(train_accuracy_listsP))
        print("Test accuracy: {}".format(test_accuracy_listsP))

        Pi = Yi.shape[0]
        
    # print("Finish constructing neural network")
    # print("Number of hidden neurons: {}.".format(n_lists))
    
    # _logger.info("Saved optimized parameters for backpropagation")
    # save_parameters(outputs, parameters_path, data, n_lists)

    return train_accuracy, test_accuracy, train_loss_GD, test_loss_GD, test_acc_GD

