# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle

def save_list(my_list, parameters_path, data):
    with open(parameters_path + data+"_"+'n_lists.json','w') as f: 
        json.dump(my_list, f, ensure_ascii=False)

def save_dic(outputs, parameters_path, data, name):
    my_file = open(parameters_path + data + "_" + name + ".pkl", "wb")
    pickle.dump(outputs, my_file)
    my_file.close()

def load_dic( parameters_path, data, name):
    my_file = open(parameters_path + data + "_" + name + ".pkl", "rb")
    output = pickle.load(my_file)
    my_file.close()
    return output

def get_batch(Y, T, index, batchSize): 
    m = Y.shape[1]
    if batchSize < m:
        if index == (round(m/batchSize)-1):
            Y_batch = Y[:, index*batchSize:]
            T_batch = T[:, index*batchSize:]
        else:
            Y_batch = Y[:, index*batchSize:(index+1)*batchSize]
            T_batch = T[:, index*batchSize:(index+1)*batchSize]
    else:
        Y_batch = Y
        T_batch = T
    return Y_batch, T_batch

def compute_cost(S, Y):    
    # S = tf.nn.softmax(S, axis=1)
    # sum_cost = tf.math.reduce_mean(tf.keras.losses.MSE(tf.transpose(Y),tf.transpose(S)))
    # sum_cost = tf.math.reduce_mean(tf.keras.losses.squared_hinge(tf.transpose(Y),tf.transpose(S)))
    sum_cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(tf.transpose(a=Y)), logits=tf.transpose(a=S)))
    return sum_cost
    
def shuffle_data(Y, T):
    indices = tf.range(start=0, limit=Y.shape[1], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_Y = tf.gather(Y, shuffled_indices, axis=1)
    shuffled_T = tf.gather(T, shuffled_indices, axis=1)
    return shuffled_Y, shuffled_T

def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_v_values(Q):
    I = np.identity(Q, dtype=np.float32)
    concatenate_I = np.concatenate((I, -I), axis=0)
    return concatenate_I

def compute_nme(S, T, m):
    """
    compute NME value 

    Parameters
    ----------
    S : np.ndarray
    predicted matrix
    T : np.ndarray
    given matrix

    Returns
    ----------
    nme : int
    NME value
    """
    numerator = norm((S - T), 'fro')
    denominator = norm(T, 'fro')
    nme = 20 * np.log10(numerator / denominator)
    return nme

def calculate_accuracy(S, T):
    # S: predicted
    # T: given
    Y = np.argmax(S, axis=0)
    T = np.argmax(T, axis=0)
    accuracy = np.sum([Y == T]) / Y.shape[0]
    return accuracy

def relu(x):
    return np.maximum(0, x)

def normalize_Z(tmp_Z, Q):
    Z_part1, Z_part2 = tmp_Z[:2*Q, :], tmp_Z[2*Q:, :]
    Z_part2 = Z_part2 / np.sum(Z_part2**2, axis=0, keepdims=True)**(1/2)
    Z = np.concatenate([Z_part1, Z_part2], axis=0)
    return Z


    
def activation(Z):
    Y = relu(Z)
    return Y

def is_higer_threshold(cNME_value, oNME_value, threshold):
    value = (oNME_value - cNME_value) / abs(oNME_value)
    is_higher = True if value >= threshold else False
    return is_higher

def compute_random_nodes_transition(Q, n_lists, delta):
    random_nodes = np.array([0])
    n_lists = np.array(n_lists) - 2 * Q
    for idx, n in enumerate(n_lists):
        if idx == 0:
            el_random_nodes = np.array([nodes for nodes in range(0, n + 1, delta)])
        else:
            el_random_nodes = np.array([nodes for nodes in range(0, n + 1, delta)]) + sum(n_lists[:idx]) - 2 * Q * len(n_lists[:idx])
        random_nodes = np.append(random_nodes, el_random_nodes)
    return random_nodes

def plot_architecture(n_lists, Q, max_n, data_path, delta):
    random_nodes_lists = np.array(n_lists) - 2 * Q
    plt.figure(figsize=(10,10))
    plt.xlabel(xlabel="Layer Number")
    plt.ylabel(ylabel="Number of random nodes")
    plt.ylim(0, max_n)
    plt.scatter(x=range(1, len(random_nodes_lists)+1), y=random_nodes_lists)
    plt.savefig(data_path +'layer_num.png')

def plot_performance(xlabel, ylabel, random_nodes, train_performances, test_performances, data_path):
    plt.figure(figsize=(10, 10))
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.plot(random_nodes, train_performances, label="Train")
    plt.plot(random_nodes, test_performances, label="Test")
    plt.legend()
    plt.savefig(data_path + ylabel + '.png')

def plot_data(Q, n_lists, train_NME_lists, test_NME_lists, train_accuracy_lists, test_accuracy_lists, data, ssfn_hparameters):
    # define variables
    delta = ssfn_hparameters["delta"]
    max_n = ssfn_hparameters["max_n"]
    figure_path = "./figure/"
    data_path = figure_path + data +"/"

    # Create some directories for preservation
    create_directory(figure_path)
    create_directory(data_path)

    # The relation between layer number and number of nodes
    plot_architecture(n_lists, Q, max_n, data_path, delta)

    # The relations between number of random nodes and performances
    random_nodes = compute_random_nodes_transition(Q, n_lists, delta)
    plot_performance("Total number of random nodes","NME", random_nodes, train_NME_lists, test_NME_lists, data_path)
    plot_performance("Total number of random nodes" ,"Accuracy", random_nodes, train_accuracy_lists, test_accuracy_lists, data_path)
    
def plot_data_backprop(iteration_num, train_accuracy_lists, test_accuracy_lists, train_NME_lists, test_NME_lists, data, learning_rate):
    # define variables
    figure_path = "./figure/"
    data_path = figure_path + data +"/"

    # Create some directories for preservation
    create_directory(figure_path) 
    create_directory(data_path)
    # The relations between number of iteration and performances
    plot_performance("Number of iteration", "NME_bp", range(0, iteration_num + 1), train_NME_lists, test_NME_lists, data_path)
    plot_performance("Number of iteration", "Accuracy_bp", range(0, iteration_num + 1), train_accuracy_lists, test_accuracy_lists, data_path)