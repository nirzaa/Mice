import os
import numpy as np
import gin
import mice
import torch
from itertools import combinations_with_replacement
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


@gin.configurable
def entropy_runner(num_boxes, max_epochs, genom, lr, weight_decay, batch_size, freq_print):
    '''
    Running the neural network in order to calculate the entropy of our system
    
    num_boxes: the number of boxes to split our space to
    max_epochs: the maximum number of epochs to use in the beginning
    genom: the type of architecture we are going to use in the neural net
    lr: the learning rate
    weight_decay: regularization technique by adding a small penalty
    batch_size: the size of the batch
    freq_print: the number of epochs between printing to the user the mutual information
    
    return:
    None
    '''
    weights_path = os.path.join('./', 'model_weights')
    PATH = os.path.join(weights_path, genom+'_model_weights.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R = np.random.RandomState(seed=1)

    num_frames = mice.frames()
    cntr = 0
    mi_entropy_dependant = []
    mi_entropy_dependant_valid = []
    my_root = int(np.floor(np.log2(num_boxes)))
    my_combinations = list(combinations_with_replacement([2 << expo for expo in range(0, my_root)], 3))
    my_combinations.sort(key=mice.sort_func)
    mice.print_combinations(my_combinations)
    number_combinations = len(my_combinations)
    x_labels = []
    for i, j, k in my_combinations:
        axis = int(np.argmax((i, j, k)))
        print('='*50)
        print(f'The size of the small boxes is: {i}x{j}x{k}\n'
              f'Therefore we cut on the {axis} axis\n'
              f'Building the boxes... we are going to start training...')
        epochs = (max_epochs // (cntr+1))
        epochs = int(np.ceil((max_epochs * 2) // ((i*j*k)**(1/3))))
        epochs = max(epochs, 1)
        sizes = (i, j, k)
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
            
            if train_losses == 'problem' or valid_losses == 'problem':
                return 'problem'
            if epoch % freq_print == 0:
                print(f'\nMI for train {train_losses[-1]}, val {valid_losses[-1]} at step {epoch}')
        cntr += 1
        x_labels.append(str((i, j, k)))
        torch.save(model.state_dict(), PATH)
        mice.entropy_fig(num=cntr, genom=genom, sizes=sizes, train_losses=train_losses, valid_losses=valid_losses)
        mice.logger(f'The MI train for ({i}, {j}, {k}) box is: {train_losses[-1]:.2f}', number_combinations=number_combinations, flag_message=1, num_boxes=num_boxes)
        mi_entropy_dependant.append(train_losses[-1])
        mi_entropy_dependant_valid.append(valid_losses[-1])
        mice.entropy_fig_running(x_labels=x_labels[:i+1], mi_entropy_dependant=mi_entropy_dependant, mi_entropy_dependant_valid=mi_entropy_dependant_valid, genom=genom)
    mice.entropy_fig_together(x_labels=x_labels, mi_entropy_dependant=mi_entropy_dependant, mi_entropy_dependant_valid=mi_entropy_dependant_valid, genom=genom)
    mi_entropy_dependant = np.array(mi_entropy_dependant)
    mice.logger(f'The total MI train is: {mi_entropy_dependant.sum():.2f}', number_combinations=number_combinations, flag_message=1)
    return None


if __name__ == '__main__':
    mice.entropy_runner()
