# Mice Package

Mice package give us the opportunity to calculate mutual information on the data.
In this file we will explain the structure of the project and how to use it.

## Table of Contents

[TOC]

### bin folder

Running `load_data` will take the `dump.0` file and turns it into `data.h5` file that the following program will use.

### data folder

The `dump.0` will be placed here and the `data.h5` file will be saved to here.

### figures folder

The figures will be saved to this folder.

### mice folder

#### main folder

Here will be the main program that we will run.

#### neural_net

The architectures of the neural nets will be saved to here

#### test folder

Here we will put our test files

#### utils folder

All our functions that we are going to use inside the `main folder` will be placed here.

### model_weights

The weights of our models will be saved to here.

## How to use the code

All the configurations are inside the `config,gin` file.
Then after generate the `data.h5` as was mentioned above,
one can run the  `box_menu` code which will take care of the rest.  

Inside the mice folder we have the `config.gin` file. In this file we are able to declare the hyperparameters and other stuff before each run as followed:  
1. inside mice.main.box_menu:
  1. boxes_sizes: the number of boxes in each axis to split our space into
  2. max_epochs: the maximum number of epochs to use in the beginning
  3. batch_size: the size of the batch
  4. freq_print: the number of epochs between printing to the user the mutual information
  5. axis: the axis we will split our boxes into, in order to calculate the mutual information
  6. genom: the type of architecture we are going to use in the neural net
  7. lr: the learning rate
  8. weight_decay: regularization technique by adding a small penalty
  8. box_frac: the fraction we will take our small box to calculate the mutual information on
2. inside mice.utils.my_utils:
  1. num_samples: number of samples we will have in each epoch
  2. samples_per_snapshot:  the number of samples to take from each snapshot
  3. flag: 0 - from data.h5 || 1 - random || 2 - log2 
  4. mod: mod 0 : prints both || mod 1 : prints only output || mod 2 : prints only to file
  5. figsize: the size of the figure
3. inside bin.load_data:
  1. number_lines: the number of lines to read from the `data.h5` file
4. inside mice.main.entropy_menu:
   1. num_boxes: the number of boxes we chose to split our space to
   2. max_epochs: the maximum number of epochs we are starting from
   3. genom: the type of the architecture for our neural net
   4. lr: the learning rate for our neural net
   5. weight_decay: regularization technique by adding a small penalty
   6. batch_size: the size of the batch
   7. freq_print: the number of epochs between printing to the user the mutual information
5. to be continued..??
6. 