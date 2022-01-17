import sys
import os
import numpy as np
my_path = os.path.join('./')
sys.path.append(my_path)
import gin
import mice
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

@gin.configurable
def box_runner(box_sizes, max_epochs, batch_size, freq_print, axis, genom, lr, weight_decay):
    weights_path = os.path.join('./','model_weights')
    PATH = os.path.join(weights_path, genom+'_model_weights.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R = np.random.RandomState(seed=0)

    num_frames = mice.frames()
    for i, num_boxes in enumerate(box_sizes):
        
        sizes = mice.sizer(num_boxes=num_boxes)
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

        torch.save(model.state_dict(), PATH)
        mice.func_fig(num=0, genom=genom, num_boxes=num_boxes, train_losses=train_losses, valid_losses=valid_losses)


if __name__ == '__main__':
    mice.box_runner()
