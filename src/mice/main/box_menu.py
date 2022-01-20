import os
import numpy as np
import gin
import mice
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split


@gin.configurable
def box_runner(box_sizes, max_epochs, batch_size, freq_print, axis, genom, lr, weight_decay, box_frac):
    '''
    Running the neural network in order to calculate the right number of boxes to split our space into

    box_sizes: the number of boxes in each axis to split our space into
    max_epochs: the maximum number of epochs to use in the beginning
    batch_size: the size of the batch
    freq_print: the number of epochs between printing to the user the mutual information
    axis: the axis we will split our boxes into, in order to calculate the mutual information
    genom: the type of architecture we are going to use in the neural net
    lr: the learning rate
    weight_decay: regularization technique by adding a small penalty
    box_frac: what is the value of the box from the total space we are calculating the mutual information to

    return:
    None
    '''
    weights_path = os.path.join('./', 'src', 'model_weights')
    PATH = os.path.join(weights_path, genom+'_model_weights.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R = np.random.RandomState(seed=0)
    mi_num_box_dependant = []
    mi_num_box_dependant_valid = []

    num_frames = mice.frames()
    for i, num_boxes in enumerate(box_sizes):
        
        sizes = mice.sizer(num_boxes=num_boxes, box_frac=box_frac)
        x_size, y_size, z_size = sizes
        model = mice.mi_model(genom=genom, n_epochs=max_epochs, max_epochs=max_epochs)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses = []
        valid_losses = []
        
        for epoch in tqdm(range(int(max_epochs))):
            lattices = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
            left_lattices, right_lattices = mice.lattice_splitter(lattices=lattices, axis=axis)
            joint_lattices = np.concatenate((left_lattices, right_lattices), axis=axis + 1)
            right_lattices_random = right_lattices.copy()
            R.shuffle(right_lattices_random)
            product_lattices = np.concatenate((left_lattices, right_lattices_random), axis=axis + 1)
            joint_lattices, joint_valid, product_lattices, product_valid = train_test_split(joint_lattices, product_lattices,
                                                                                            test_size=0.2, random_state=42)

            AB_joint, AB_product = torch.tensor(joint_lattices), torch.tensor(product_lattices)
            AB_joint_train, AB_product_train = AB_joint.to(device), AB_product.to(device)
            dataset_train = mice.MiceDataset(x_joint=AB_joint_train, x_product=AB_product_train)

            AB_joint, AB_product = torch.tensor(joint_valid), torch.tensor(product_valid)
            AB_joint_valid, AB_product_valid = AB_joint.to(device), AB_product.to(device)
            dataset_valid = mice.MiceDataset(x_joint=AB_joint_valid, x_product=AB_product_valid)

            loader = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=0, shuffle=False)
            loss_train, mutual_train = mice.train_one_epoch(model=model, data_loader=loader, optimizer=optimizer)
            train_losses.append(mutual_train.cpu().detach().numpy())

            loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, num_workers=0, shuffle=False)
            valid_loss, valid_mutual = mice.valid_one_epoch(model=model, data_loader=loader)
            valid_losses.append(valid_mutual.cpu().detach().numpy())

            if epoch % freq_print == 0:
                print(f'\nMI for train {train_losses[-1]}, val {valid_losses[-1]} at step {epoch}')
        mice.logger(f'MI train for {num_boxes} boxes is: {train_losses[-1]:.2f}\n'
                    f'The fraction we split to is: {box_frac}, therefore the small box size is: {x_size}x{y_size}x{z_size}', number_combinations=len(box_sizes), flag_message=0)
        torch.save(model.state_dict(), PATH)
        train_losses = mice.exp_ave(data=train_losses)
        # valid_losses = mice.exp_ave(data=valid_losses)
        mice.box_fig(num=0, genom=genom, num_boxes=num_boxes, train_losses=train_losses, valid_losses=valid_losses)
        mi_num_box_dependant.append(train_losses[-1])
        mi_num_box_dependant_valid.append(valid_losses[-1])
        mice.box_fig_running(box_sizes[:i+1], mi_num_box_dependant, mi_num_box_dependant_valid, genom)
    mi_num_box_dependant = np.array(mi_num_box_dependant)
    mice.box_fig_together(box_sizes, mi_num_box_dependant, mi_num_box_dependant_valid, genom)
    return None


if __name__ == '__main__':
    mice.box_runner()
