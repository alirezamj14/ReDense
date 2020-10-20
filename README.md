# SSFN

All the materials available in this document are to reproduce the results published in the following paper:

>S. Catterjee, A. M. Javid, M. Sadeghi, S. Kikuta, P.P. Mitra, M. Skoglund, 
>"SSFN: Low Complexity Self Size-estimating Feed-forward Neural Network using Layer-wise Convex Optimization", 2019

SSFN is the method for estimating the architecture of neural network.   
The code is organized as follows:

- main.py: Govern to construct a neural network and back propagation.
- multi_layer_ssfn.py: Build a neural network.
- optimize_wl.py: Optimize the matrix W showed in the paper by solving least-square problem on 1st layer.
- optimize_output.py: Construct a neural network and optimize the matrix O on each layer by ADMM method.
- make_dataset_helper.py: Make datasets used for experiments.
- function.py: Define the helper function for all other files.

In "mat_files" folder, you find the used datasets in our experiments. 
This folder must be placed in the same directory as the codes.   

### Preparation
Before to execute SSFN, it is necessary to install some packages written in "requirement.txt".   
You may install them by executing the following command.   
```pip install -r requirement.txt```

### Basic Usage
To run SSFN on certain dataset, execute the following command.   
```python main.py --data *dataset_name*```   

For example, in order to implement SSFN on Vowel dataset based on the parameters TABLE Ⅱ shows, execute the following command.   
```python main.py --data vowel --lambda_ls 100 --myu 1000 --max_k 100 --alpha 2 --max_n 1000 --eta_n 0.005 --eta_l 0.1 --max_l 20 --delta 50 --learning_rate 0.000001 --iteration_num 1000```

It is also possible to execute the above command using the default argument like as follows.   
```python main.py --data vowel --lambda_ls 100 --myu 1000 --learning_rate 0.000001```

### Options 
You can check out the options with SSFN using:   
```python main.py --help```

########################################################################################################################
########################################################################################################################
%
%   Contact:    Saikat Chatterjee (sach@kth.se), Alireza Javid (almj@kth.se) 
%
% 	April 2019
   