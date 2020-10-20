# -*- coding: utf-8 -*-

import logging 
import argparse
from HNF import HNF
import tensorflow as tf
from MLP import MLP
from CNN import CNN
from SOTA_2 import SOTA_2
from ReBoost import ReBoost
from MyFunctions import *
from make_dataset_helper import *

def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="cifar10", help="Input dataset available as the paper shows")
    parser.add_argument("--lam", type=float, default=10**(-10), help="Reguralized parameters on the least-square problem")
    parser.add_argument("--mu", type=float, default=10**(-7), help="Parameter for ADMM")
    parser.add_argument("--kMax", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--n1", type=int, default=250, help="Max number of random nodes on each layer")
    parser.add_argument("--LayerNum", type=int, default=1, help="Parameter for ADMM")
    parser.add_argument("--learning_rate", type=float, default=10**(-6), help="Learning rate for back propagation")
    parser.add_argument("--Epoch_num", type=int, default=100, help="Iteration number of back propagation") 
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size of SGD") 
    parser.add_argument("--n1_RB", type=int, default=4000, help="Max number of random nodes on each layer")
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

def main():
    args = define_parser()
    _logger = define_logger()
    # X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    HNF_hparameters, MLP_hparameters, RB_hparameters = set_hparameters(args)
    
    # _logger.info("Construct HNF")
    # HNF(_logger, X_train, X_test, T_train, T_test, HNF_hparameters)
    
    # _logger.info("Construct MLP")
    # MLP(_logger, X_train, X_test, T_train, T_test, MLP_hparameters)

    # _logger.info("Construct CNN")
    # CNN("categorical_crossentropy","softmax")

    # _logger.info("Construct SOTA")
    # SOTA_2()

    model_name = "BiT-M-R50x1"
    # model_name = "BiT_mixup"
    # model_name = "BiT-M-R101x3"
    # model_name = "BiT-M-R101x3_few"

    _logger.info("Start ReBoost Optimization")
    RB_hparameters["architecture_name"] = model_name
    ReBoost(RB_hparameters)

    # MyOptimizers=["Adadelta","Adagrad","Adam","Adamax","Nadam","RMSprop"]
    # result_path = "./results/"
    # data=RB_hparameters["data"]
    # data_path = result_path + data + "_"

    # for name in MyOptimizers:
    #     RB_hparameters["optimizer_RB"]=name
    #     train_accuracy_lists = []
    #     test_accuracy_lists = []
    #     for value in [10**(-8),10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**(0),10**(1)]:
    #         RB_hparameters["learning_rate_RB"]=value
    #         train_accuracy, test_accuracy = ReBoost(RB_hparameters)
    #         train_accuracy_lists.append(train_accuracy)
    #         test_accuracy_lists.append(test_accuracy)
    #     plt.subplots()
    #     plt.plot(range(0, len(test_accuracy_lists)), np.array(test_accuracy_lists), 'r-', label="Test Accuracy")
    #     plt.plot(range(0, len(train_accuracy_lists)), np.array(train_accuracy_lists), 'b-', label="Train Accuracy")
    #     plt.legend(loc='best')
    #     plt.savefig(data_path +"Acc_vs_LR_"+name+"_batch_" \
    #         +str(RB_hparameters["batchSize_RB"])+"_learningRate_"+str(value)+".png")
    #     plt.close()

if __name__ == '__main__':
    main()