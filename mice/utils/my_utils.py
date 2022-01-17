import h5py
import numpy as np
import os
import gin
import torch
import mice
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MiceDataset(Dataset):
    '''
    Defining the Dataset to be used in the DataLoader

    x_joint: the lattices we got for the joint
    x_product: the lattices we got for the product
    n_samples: the number of lattices we got
    '''

    def __init__(self, x_joint, x_product):
        self.x_joint = x_joint
        self.x_product = x_product
        self.n_samples = x_joint.shape[0]

    def __getitem__(self, item):
        return self.x_joint[item], self.x_product[item]

    def __len__(self):
        return self.n_samples

def read_data():
    '''
    Reading the data to train our neural net on

    return:
    blocks: the coordinates of our particles in different frames in time
    '''
    my_hf = h5py.File(os.path.join('./','data','data.h5'), 'r')
    n1 = my_hf.get('dataset_1')
    blocks = np.array(n1)
    my_hf.close()
    return blocks

def frames():
    '''
    Calculating the number of frames in the input

    return:
    the number of frames
    '''
    blocks = read_data()
    num_frames = blocks.shape[0]  # number of frames in the input
    print(f'The number of frames in the input is: {num_frames}',
    f'\n',
    '='*50)
    return num_frames

@gin.configurable
def sizer(num_boxes, box_frac):
    '''
    Calculate the size for our boxes to split our space to
    
    num_boxes: the number of boxes we split our space to
    box_frac: what is the value of the box from the total space we are calculating the mutual information to
    
    return:
    the size of our box we are calculating the mutual information to
    '''
    x_size, y_size, z_size = int(np.floor(num_boxes*box_frac)), int(np.floor(num_boxes*box_frac)), int(np.floor(num_boxes*box_frac))
    x_size, y_size, z_size = x_size - x_size%2, y_size - y_size%2, z_size - z_size%2
    print(f'\nWe split the space into {num_boxes}x{num_boxes}x{num_boxes} boxes\n',
        f'The size of the small box is: ({x_size}, {y_size}, {z_size})\n',
        f'='*50)
    return (x_size, y_size, z_size)

def mi_model(genom, n_epochs, max_epochs):
    '''
    Declare the model and loading the weights if necessary
    
    genom: the type of architecture we will use for the neural net
    n_epochs: number of epochs in the current run
    max_epochs: the maximum number of epochs we are using at the start, before we are using transfer learning

    return:
    the relevant model loaded with its weights
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"
    weights_path = os.path.join('./','model_weights')

    if genom == 'linear':
        model = mice.Net()
        model.to(device)
    elif genom == 'fcn':
        model = mice.Model()
        model.to(device)
    elif genom == 'new_fcn':
        model = mice.Modely()
        model.to(device)

    if n_epochs != max_epochs and genom == 'linear':
        print(f'==== linear ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'lin_model_weights.pth')
        model = mice.Net()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'linear':
        PATH = os.path.join(weights_path, 'lin_model_weights.pth')
        print(f'==== linear ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'fcn':
        print(f'==== fcn ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'fcn_model_weights.pth')
        model = mice.Model()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'fcn':
        PATH = os.path.join(weights_path, 'fcn_model_weights.pth')
        print(f'==== fcn ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'new_fcn':
        print(f'==== new fcn ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'new_fcn_model_weights.pth')
        model = mice.Modely()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'new_fcn':
        PATH = os.path.join(weights_path, 'new_fcn_model_weights.pth')
        print(f'==== new fcn ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    return model

@gin.configurable
def boxes_maker(num_boxes, sample, flag):
    '''
    Generating the sliced box of the sample mentioned in {sample}
    
    num_boxes: the number of boxes we split our space to
    sample: number of sample we chose randomly to take our data from
    flag: whether to use the data from data.h5 (flag=0), random data (flag=1), or the data that reproduces log2 mutual information (flag=2)

    return:
    our splitted space into the number of boxes we defined
    '''
    if flag == 0:
        borders = np.linspace(0, 1, num_boxes+1, endpoint=True)
        blocks = read_data()
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))

        df_particles = pd.DataFrame(
            blocks[sample],
            columns=['X', 'Y', 'Z']
        )

        x_bin = borders.searchsorted(df_particles['X'].to_numpy())
        y_bin = borders.searchsorted(df_particles['Y'].to_numpy())
        z_bin = borders.searchsorted(df_particles['Z'].to_numpy())

        g = dict((*df_particles.groupby([x_bin, y_bin, z_bin]),))

        g_keys = list(g.keys())

        for cntr, cor in enumerate(g_keys):
            boxes_tensor[cor[0]-1, cor[1]-1, cor[2]-1] = 1

    elif flag == 1:
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
        flag_torch = torch.randint_like(torch.tensor(boxes_tensor), low=0, high=2)
        boxes_tensor[flag_torch == 1] = 1

    elif flag == 2:
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
        i = np.random.randint(low=0, high=boxes_tensor.shape[0])
        j = np.random.randint(low=0, high=boxes_tensor.shape[1])
        k = np.random.randint(low=0, high=boxes_tensor.shape[2])
        boxes_tensor[i, j, k] = 1
    
    return boxes_tensor

@gin.configurable
def lattices_generator(num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes, cntr=0, lattices=None):
    '''
    Generate the lattices that will be used in our neural net
    
    num_samples: number of samples we will have in each epoch
    samples_per_snapshot: the number of samples to take from each snapshot
    R: np.random.RandomState
    num_frames: number of frames we have in the data, from it we will pick 1 frame randomly to take our data from
    num_boxes: the number of boxes we split our space to
    sizes: the sizes of the box we are calculating the mutual information to
    cntr: just a counter
    lattice: a list we will put the lattices we will construct into

    return:
    list of lattices we've constracted
    '''
    if lattices is None:
        lattices = []
    x_size, y_size, z_size = sizes
    num_sample = R.randint(num_frames)
    my_tensor = mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
    leny = my_tensor.shape[0]
    x_steps = leny - x_size
    y_steps = leny - y_size
    z_steps = leny - z_size
    while True:
        if x_steps == 0:
            i = 0
        else:
            i = R.randint(0, x_steps+1)
        if y_steps == 0:
            j = 0
        else:
            j = R.randint(0, y_steps+1)
        if z_steps == 0:
            k = 0
        else:
            k = R.randint(0, z_steps+1)

        lattices.append(my_tensor[i:i+x_size, j:j+y_size, k:k + z_size])
        cntr += 1
        if cntr == num_samples:
            return lattices
        elif cntr % (samples_per_snapshot+1) == 0:
            return lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes, cntr=cntr, lattices=lattices)

def lattice_splitter(lattices, axis):
    '''
    Here we are splitting the lattices given in {lattices} on the axis given in the {axis}
    
    lattices: list of the lattices we've constructed
    axis: the axis we will split our lattices on

    return:
    left lattices and right lattices
    '''
    
    left_lattices, right_lattices = [], []
    for lattice in lattices:
        left_lattice, right_lattice = np.split(lattice, 2, axis=axis)
        left_lattices.append(left_lattice)
        right_lattices.append(right_lattice)
    return np.array(left_lattices), np.array(right_lattices)

def loss_function(joint_output, product_output):
    """
    calculating the loss function
    
    joint_output: the joint lattices we've constructed
    product_output: the product lattices we've constructed

    return:
    mutual: the mutual information
    joint_output: the joint lattices we've constructed
    exp_product: the exponent of the product_output 
    """
    exp_product = torch.exp(product_output)
    mutual = torch.mean(joint_output) - torch.log(torch.mean(exp_product))
    return mutual, joint_output, exp_product

def train_one_epoch(model, data_loader, optimizer, ma_rate=0.01):
    '''
    train one epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    optimizer: optimizer
    ma_rate: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    model.train()
    total_loss = 0
    total_mutual = 0
    for batch_idx, data in enumerate(data_loader):
        loss, mutual = train_one_step(model, data, optimizer, ma_rate)
        total_loss += loss
        total_mutual += mutual
    return total_loss / len(data_loader), total_mutual / len(data_loader)
def train_one_step(model, data, optimizer, ma_rate, ma_et=1.0):
    '''
    train one batch in the epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    optimizer: optimizer
    ma_rate: used in order to calculate the loss function
    ma_et: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    x_joint, x_product = data
    optimizer.zero_grad()
    joint_output = model(x_joint.float())
    product_output = model(x_product.float())
    try:
        mutual, joint_output, exp_product = loss_function(joint_output, product_output)
    except Exception as e:
        print(f'The error is:\n{e}')
        return 'problem'
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_product)
    loss_train = -(torch.mean(joint_output) - (1 / ma_et.mean()).detach() * torch.mean(exp_product))
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    return loss_train, mutual

def valid_one_epoch(model, data_loader, ma_rate=0.01):
    '''
    validation of one epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    ma_rate: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    model.eval()
    total_loss = 0
    total_mutual = 0
    for batch_idx, data in enumerate(data_loader):
        with torch.no_grad():
            loss, mutual = valid_one_step(model, data, ma_rate)
        total_loss += loss
        total_mutual += mutual
    return total_loss / len(data_loader), total_mutual / len(data_loader)
def valid_one_step(model, data, ma_rate, ma_et=1.0):
    '''
    validation of one batch in the epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    ma_rate: used in order to calculate the loss function
    ma_et: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    x_joint, x_product = data
    joint_output = model(x_joint.float())
    product_output = model(x_product.float())
    try:
        mutual, joint_output, exp_product = loss_function(joint_output, product_output)
    except Exception as e:
        print(f'The error is:\n{e}')
        return 'problem'
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_product)
    loss_train = -(torch.mean(joint_output) - (1 / ma_et.mean()).detach() * torch.mean(exp_product))
    return loss_train, mutual

def func_fig(num, genom, num_boxes, train_losses, valid_losses):
    '''
    Plot and print the results
    
    num: number of the figure
    genom: the type of architecture we've trained our neural net with
    num_boxes: the number of boxes we split our space to
    train_losses: the losses we got from training
    valid_losses: the losses we got from validating

    return:
    mi_train: mutual information gained from training
    mi_valid: mutual information gained from validating
    '''
    plt.figure(num=num)
    plt.title(f'Searching for perfect box number, try: {num_boxes} boxes')
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    saved_path = os.path.join('./', 'figures', "losses", "box_size_search", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, str(num_boxes)+"_boxes"))
    plt.figure(num=num).clear()
    plt.close(num)
    mi_train, mi_valid = train_losses[-1], valid_losses[-1]
    print(f'The MI train for {num_boxes} boxes is: {mi_train}')
    return mi_train, mi_valid

def folder_checker(path):
    '''
    if a folder in {path} does not exist, this function will create it

    return:
    None
    '''
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    return None
        