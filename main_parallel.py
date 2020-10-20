# -*- coding: utf-8 -*-

import logging 
import argparse
from HNF import HNF
import tensorflow as tfs
from MLP import MLP
from CNN import CNN
from ReBoost import ReBoost
from MyFunctions import *
from make_dataset_helper import *
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="MNIST", help="Input dataset available as the paper shows")
    parser.add_argument("--lam", type=float, default=10**(-10), help="Reguralized parameters on the least-square problem")
    parser.add_argument("--mu", type=float, default=10**(-7), help="Parameter for ADMM")
    parser.add_argument("--kMax", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--n1", type=int, default=250, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum", type=int, default=1, help="Parameter for ADMM")
    parser.add_argument("--learning_rate", type=float, default=10**(-6), help="Learning rate for back propagation")
    parser.add_argument("--Epoch_num", type=int, default=100, help="Iteration number of back propagation") 
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size of SGD") 
    parser.add_argument("--n1_RB", type=int, default=500, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum_RB", type=int, default=1, help="Parameter for ADMM")
    parser.add_argument("--learning_rate_RB", type=float, default=10**(-4), help="Learning rate for back propagation")
    parser.add_argument("--Epoch_num_RB", type=int, default=100, help="Epoch number of back propagation") 
    parser.add_argument("--batchSize_RB", type=int, default=50000, help="Batch size of SGD") 
    parser.add_argument("--optimizer_RB", default="Adam", help="Gradient descent optimizer")
    parser.add_argument("--initialization", default="Off", help="Unit variance random initialization")
    args = parser.parse_args()
    return args

    
def define_logger():
    _logger = logging.getLogger(__name__)
    logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
    return _logger

def define_dataset(args):
    if args.data == "letter":
        X_train,X_test, T_train,  T_test  = prepare_letter()
    elif args.data == "shuttle":
        X_train,X_test, T_train,  T_test  = prepare_shuttle()
    elif args.data == "MNIST":
        X_train,X_test, T_train,  T_test  = prepare_mnist()
    return X_train, X_test, T_train, T_test

def set_hparameters(args):
    HNF_hparameters = {"data": args.data, "lam": args.lam, "mu": args.mu, \
            "kMax": args.kMax, "n1": args.n1, "LayerNum": args.LayerNum}
    MLP_hparameters = {"data": args.data, "Epoch_num": args.Epoch_num, \
            "batchSize": args.batchSize, "learning_rate": args.learning_rate,\
                 "LayerNum": args.LayerNum}
    RB_hparameters = {"data": args.data, "Epoch_num_RB": args.Epoch_num_RB, \
            "learning_rate_RB": args.learning_rate_RB, "n1_RB": args.n1_RB, \
                "batchSize_RB": args.batchSize_RB, "LayerNum_RB": args.LayerNum_RB, \
                    "optimizer_RB": args.optimizer_RB, "initialization": args.initialization}
    return HNF_hparameters, MLP_hparameters, RB_hparameters

def crossVal_ReBoost(learning_rate,RB_hparameters):
    RB_hparameters["learning_rate_RB"]=learning_rate
    train_accuracy, test_accuracy, _, _, _ = ReBoost(RB_hparameters)
    return (train_accuracy, test_accuracy)

def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    HNF_hparameters, MLP_hparameters, RB_hparameters = set_hparameters(args)
    
    _logger.info("Construct HNF")
    HNF(_logger, X_train, X_test, T_train, T_test, HNF_hparameters)
    
    _logger.info("Construct MLP")
    _logger.info("Learning rate={}".format(args.learning_rate))
    MLP(_logger, X_train, X_test, T_train, T_test, MLP_hparameters)

    _logger.info("Start ReBoost Optimization")
    # ReBoost(_logger, RB_hparameters)

if __name__ == '__main__':
    # main()
    args = define_parser()
    _, _, RB_hparameters = set_hparameters(args)
    parameters_path = "./parameters/"
    result_path = "./results/"
    data = RB_hparameters["data"]
    data_path = result_path + data + "_"

    ############################################################################
    ############    For implementing on MLP with different neurons n  ##########
    ############################################################################

    # num_cores = multiprocessing.cpu_count()
    MyOptimizer="Adam"#,"Adagrad","Adam","Adamax","Nadam","RMSprop"]
    batchSize=60000
    neurons = [500,1000,1500,2000]
    Epoch_num = 300
    flag = "Off"
    learning_rate_list = [10**(-5)]
    train_loss_dic = {}
    test_loss_dic = {}
    test_acc_dic = {}
    train_loss_dic = load_dic( parameters_path, data, "TrainLoss_GD_MLP")
    test_loss_dic = load_dic( parameters_path, data, "TestLoss_GD_MLP")
    test_acc_dic = load_dic( parameters_path, data, "TestAcc_GD_MLP")
    for neuron in neurons:
        for learning_rate in learning_rate_list:
            RB_hparameters["architecture_name"] = "MLP"
            RB_hparameters["learning_rate_RB"]=learning_rate
            RB_hparameters["batchSize_RB"]=batchSize
            RB_hparameters["optimizer_RB"]=MyOptimizer
            RB_hparameters["Epoch_num_RB"]=Epoch_num
            RB_hparameters["n1_RB"] = neuron
            RB_hparameters["initialization"] = flag
            print(neuron)
            
            _, _, train_loss_GD, test_loss_GD, test_acc_GD = ReBoost(RB_hparameters)

            train_loss_dic[str(neuron)]=train_loss_GD
            test_loss_dic[str(neuron)]=test_loss_GD
            test_acc_dic[str(neuron)]=test_acc_GD

            # train_loss_dic[flag]=train_loss_GD
            # test_loss_dic[flag]=test_loss_GD
            # test_acc_dic[flag]=test_acc_GD

    # save_dic(train_loss_dic, parameters_path, data, "TrainLoss_GD_MLP")
    # save_dic(test_loss_dic, parameters_path, data, "TestLoss_GD_MLP")
    # save_dic(test_acc_dic, parameters_path, data, "TestAcc_GD_MLP")

            # csfont = {'fontname':'sans-serif'}
            # plt.subplots()
            # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["On"], 'b-', label="with")
            # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["Off"], 'r-', label="without")
            # plt.legend(loc='best')
            # plt.xlabel("Number of epochs",fontdict=csfont)
            # plt.ylabel("Training loss",fontdict=csfont)
            # plt.title("MNIST, Multilayer Neural Network", loc='center')
            # plt.savefig(result_path + "MLP_TrainLoss_"+str(learning_rate)+"_2.png")
            # plt.close()

            # plt.subplots()
            # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["On"], 'b-', label="with")
            # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["Off"], 'r-', label="without")
            # plt.legend(loc='best')
            # plt.xlabel("Number of epochs",fontdict=csfont)
            # plt.ylabel("Testing loss",fontdict=csfont)
            # plt.title("MNIST, Multilayer Neural Network", loc='center')
            # plt.savefig(result_path + "MLP_TestLoss_"+str(learning_rate)+"_2.png")
            # plt.close()

            # plt.subplots()
            # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_acc_dic["On"], 'b-', label="with")
            # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_acc_dic["Off"], 'r-', label="without")
            # plt.legend(loc='best')
            # plt.xlabel("Number of epochs",fontdict=csfont)
            # plt.ylabel("Testing Accuracy",fontdict=csfont)
            # plt.title("MNIST, Multilayer Neural Network", loc='center')
            # plt.savefig(result_path + "MLP_TestAcc_"+str(learning_rate)+"_2.png")
            # plt.close()



    # FontSize = 16
    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic[str(500)], 'b-', label="m = 500", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic[str(1000)], 'r-', label="m = 1000", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic[str(1500)], 'g-', label="m = 1500", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic[str(2000)], 'm-', label="m = 2000", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Training loss",fontdict=csfont, fontsize=FontSize)
    # # plt.title("MNIST, Multilayer Neural Network", loc='center', fontsize=FontSize)
    # plt.grid()
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path + "MLP_TrainLoss.png", dpi=600)
    # plt.close()

    # plt.subplots()
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic[str(500)], 'b-', label="m = 500", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic[str(1000)], 'r-', label="m = 1000", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic[str(1500)], 'g-', label="m = 1500", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic[str(2000)], 'm-', label="m = 2000", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Testing loss",fontdict=csfont, fontsize=FontSize)
    # # plt.title("MNIST, Multilayer Neural Network", loc='center', fontsize=FontSize)
    # plt.grid()
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path + "MLP_TestLoss.png", dpi=600)
    # plt.close()

    # test_acc_dic_500 = np.asarray(test_acc_dic[str(500)])
    # test_acc_dic_1000 = np.asarray(test_acc_dic[str(1000)])
    # test_acc_dic_1500 = np.asarray(test_acc_dic[str(1500)])
    # test_acc_dic_2000 = np.asarray(test_acc_dic[str(2000)])

    # plt.subplots()
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic[str(500)]) * 100, 'b-', label="m = 500", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic[str(1000)]) * 100, 'r-', label="m = 1000", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic[str(1500)]) * 100, 'g-', label="m = 1500", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic[str(2000)]) * 100, 'm-', label="m = 2000", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Testing Accuracy (%)",fontdict=csfont, fontsize=FontSize)
    # # plt.title("MNIST, Multilayer Neural Network", loc='center', fontsize=FontSize)
    # plt.grid()
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path + "MLP_TestAcc.png", dpi=600)
    # plt.close()

    ############################################################################
    ########    For implementing on CNN with different loss functions  #########
    ############################################################################

    # num_cores = multiprocessing.cpu_count()
    # MyOptimizer="Adam"#,"Adagrad","Adam","Adamax","Nadam","RMSprop"]
    # batchSize=50000
    # learning_rate = 10**(-4)
    # RB_hparameters["n1_RB"] = 512
    # data = RB_hparameters["data"]
    # Loss_list = ["categorical_crossentropy","MSE","poisson","Huber"]
    # activation_list = ["softmax"]#,"sigmoid","softplus","exponential"] 
    # train_loss_dic = {}
    # test_loss_dic = {}
    # test_acc_dic = {}
    # train_loss_dic = load_dic( parameters_path, data, "TrainLoss_GD_CNN")
    # test_loss_dic = load_dic( parameters_path, data, "TestLoss_GD_CNN")
    # test_acc_dic = load_dic( parameters_path, data, "TestAcc_GD_CNN")
    # for Myactivation in activation_list:
    #     for Myloss in Loss_list:
    #         print(Myloss)
    #         # CNN(Myloss,Myactivation,data)

    #         RB_hparameters["learning_rate_RB"]=learning_rate
    #         RB_hparameters["batchSize_RB"]=batchSize
    #         RB_hparameters["optimizer_RB"]=MyOptimizer
    #         RB_hparameters["architecture_name"] = "CNN_"+Myloss+"_"+Myactivation
            
            
    #         # _, _, train_loss_GD, test_loss_GD, test_acc_GD = ReBoost(RB_hparameters)

    #         # train_loss_dic[Myactivation]=train_loss_GD
    #         # test_loss_dic[Myactivation]=test_loss_GD
    #         # test_acc_dic[Myactivation]=test_acc_GD

    #         # train_loss_dic[Myloss]=train_loss_GD
    #         # test_loss_dic[Myloss]=test_loss_GD
    #         # test_acc_dic[Myloss]=test_acc_GD

    # save_dic(train_loss_dic, parameters_path, data, "TrainLoss_GD_CNN")
    # save_dic(test_loss_dic, parameters_path, data, "TestLoss_GD_CNN")
    # save_dic(test_acc_dic, parameters_path, data, "TestAcc_GD_CNN")


    # FontSize = 16
    # # csfont = {'fontname':'sans-serif'}
    # # plt.subplots()
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["softmax"], 'b-', label="softmax", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["sigmoid"], 'r-', label="sigmoid", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["softplus"], 'g-', label="softplus", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["exponential"], 'm-', label="exponential", linewidth=2)
    # # plt.legend(loc='best', fontsize=FontSize)
    # # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # # plt.ylabel("Training loss",fontdict=csfont, fontsize=FontSize)
    # # plt.title("CIFAR-10, Convolutional Neural Network", loc='center', fontsize=FontSize)
    # # # plt.savefig(data_path +"TrainLoss_vs_epoch_"+MyOptimizer+"_batch_" \
    # # #     +str(RB_hparameters["batchSize_RB"])+"_epoch_"+str(RB_hparameters["Epoch_num_RB"])+".png")
    # # plt.grid()
    # # plt.xticks(fontsize=FontSize)
    # # plt.yticks(fontsize=FontSize)
    # # plt.tight_layout()
    # # plt.savefig(result_path + "CNN_TrainLoss_activations.png", dpi=600)    
    # # plt.close()

    # # plt.subplots()
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["softmax"], 'b-', label="softmax", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["sigmoid"], 'r-', label="sigmoid", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["softplus"], 'g-', label="softplus", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["exponential"], 'm-', label="exponential", linewidth=2)
    # # plt.legend(loc='best', fontsize=FontSize)
    # # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # # plt.ylabel("Testing loss",fontdict=csfont, fontsize=FontSize)
    # # plt.title("CIFAR-10, Convolutional Neural Network", loc='center', fontsize=FontSize)
    # # plt.grid()
    # # plt.xticks(fontsize=FontSize)
    # # plt.yticks(fontsize=FontSize)
    # # plt.tight_layout()
    # # plt.savefig(result_path + "CNN_TestLoss_activations.png", dpi=600)
    # # plt.close()

    # # plt.subplots()
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["softmax"])*100, 'b-', label="softmax", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["sigmoid"])*100, 'r-', label="sigmoid", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["softplus"])*100, 'g-', label="softplus", linewidth=2)
    # # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["exponential"])*100, 'm-', label="exponential", linewidth=2)
    # # plt.legend(loc='best', fontsize=FontSize)
    # # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # # plt.ylabel("Testing Accuracy",fontdict=csfont, fontsize=FontSize)
    # # plt.title("CIFAR-10, Convolutional Neural Network", loc='center', fontsize=FontSize)
    # # plt.grid()
    # # plt.xticks(fontsize=FontSize)
    # # plt.yticks(fontsize=FontSize)
    # # plt.tight_layout()
    # # plt.savefig(result_path + "CNN_TestAcc_activations.png", dpi=600)
    # # plt.close()



    # csfont = {'fontname':'sans-serif'}
    # plt.subplots()
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["categorical_crossentropy"], 'b-', label="CE", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["MSE"], 'r-', label="MSE", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["poisson"], 'g-', label="Poisson", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), train_loss_dic["Huber"], 'm-', label="Huber", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Training loss",fontdict=csfont, fontsize=FontSize)
    # # plt.title("CIFAR-10, Convolutional Neural Network", loc='center', fontsize=FontSize)
    # # plt.savefig(data_path +"TrainLoss_vs_epoch_"+MyOptimizer+"_batch_" \
    # #     +str(RB_hparameters["batchSize_RB"])+"_epoch_"+str(RB_hparameters["Epoch_num_RB"])+".png")
    # plt.grid()
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path + "CNN_TrainLoss.png", dpi=600)    
    # plt.close()

    # plt.subplots()
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["categorical_crossentropy"], 'b-', label="CE", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["MSE"], 'r-', label="MSE", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["poisson"], 'g-', label="Poisson", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), test_loss_dic["Huber"], 'm-', label="Huber", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Testing loss",fontdict=csfont, fontsize=FontSize)
    # # plt.title("CIFAR-10, Convolutional Neural Network", loc='center', fontsize=FontSize)
    # plt.grid()
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path + "CNN_TestLoss.png", dpi=600)
    # plt.close()

    # plt.subplots()
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["categorical_crossentropy"])*100, 'b-', label="CE", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["MSE"])*100, 'r-', label="MSE", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["poisson"])*100, 'g-', label="Poisson", linewidth=2)
    # plt.plot(range(0, RB_hparameters["Epoch_num_RB"]), np.asarray(test_acc_dic["Huber"])*100, 'm-', label="Huber", linewidth=2)
    # plt.legend(loc='best', fontsize=FontSize)
    # plt.xlabel("Number of epochs",fontdict=csfont, fontsize=FontSize)
    # plt.ylabel("Testing Accuracy",fontdict=csfont, fontsize=FontSize)
    # # plt.title("CIFAR-10, Convolutional Neural Network", loc='center', fontsize=FontSize)
    # plt.grid()
    # plt.xticks(fontsize=FontSize)
    # plt.yticks(fontsize=FontSize)
    # plt.tight_layout()
    # plt.savefig(result_path + "CNN_TestAcc.png",dpi=600)
    # plt.close()


    ############################################################################
    ########################    For cross validation on learning rate   ########
    ############################################################################

    # num_cores = multiprocessing.cpu_count()
    # MyOptimizers=["Adam"]#["Adadelta","Adagrad","Adam","Adamax","Nadam","RMSprop"]
    # batchSizes=[50000]
    # neuron = 500
    # Epoch_num = 100
    # CV_learning_rate = {}
    # train_loss_dic = {}
    # test_loss_dic = {}
    # test_acc_dic = {}
    # for MyOptimizer in MyOptimizers:
    #     for batchSize in batchSizes:
    #         RB_hparameters["initialization"] = "On"
    #         RB_hparameters["batchSize_RB"]=batchSize
    #         RB_hparameters["optimizer_RB"]=MyOptimizer
    #         RB_hparameters["Epoch_num_RB"]=Epoch_num
    #         RB_hparameters["n1_RB"] = neuron
            
    #         sweep = [10**(-8),10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**(0),10**(1)]
    #         Len = len(sweep)
    #         outputs = Parallel(n_jobs=20)\
    #             (delayed(crossVal_ReBoost)(i,RB_hparameters) for i in sweep)

    #         train_accuracy_lists = []
    #         test_accuracy_lists = []
    #         for output in outputs:
    #             train_accuracy, test_accuracy = output
    #             train_accuracy_lists.append(train_accuracy)
    #             test_accuracy_lists.append(test_accuracy)

    #         CV_learning_rate[MyOptimizer] = sweep[test_accuracy_lists.index(max(test_accuracy_lists))]

    #         print(MyOptimizer+" with batch "+ str(batchSize) +" learning rate "+ \
    #              str(CV_learning_rate[MyOptimizer]) +" : "+str(max(test_accuracy_lists)))

    #         plt.subplots()
    #         plt.plot(range(0, Len), test_accuracy_lists, 'r-', label="Test Accuracy")
    #         plt.plot(range(0, Len), train_accuracy_lists, 'b-', label="Train Accuracy")
    #         plt.legend(loc='best')
    #         plt.savefig(data_path +"Acc_vs_LR_"+MyOptimizer+"_batch_"+str(batchSize)+"_epoch_100.png")
    #         plt.close()
    
    # save_dic(CV_learning_rate, parameters_path, data, "_CV_lr_CNN")