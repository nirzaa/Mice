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