import mice
import numpy as np
import gin
from tqdm import tqdm
import pandas as pd
import os
from itertools import combinations_with_replacement
import h5py

def lat_saver(num_samples, samples_per_snapshot):
    R = np.random.RandomState(seed=0)
    num_frames = mice.frames()
    num_boxes = 10
    my_root = int(np.floor(np.log2(num_boxes)))
    my_combinations = list(combinations_with_replacement([2 << expo for expo in range(0, my_root)], 3))
    my_combinations.sort(key=mice.sort_func)
    for sizes in my_combinations:
        lattices = []
        x_size, y_size, z_size = sizes
        num_sample = R.randint(num_frames)
        my_tensor = mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
        leny_x = my_tensor.shape[0]
        leny_y = my_tensor.shape[1]
        leny_z = my_tensor.shape[2]
        x_steps = leny_x - x_size
        y_steps = leny_y - y_size
        z_steps = leny_z - z_size
        cntr = 0

        borders = np.linspace(0, 1, num_boxes+1, endpoint=True)
        blocks = mice.read_data()
        my_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
        num_sample = R.randint(num_frames)
        df_particles = pd.DataFrame(
        blocks[num_sample],
        columns=['X', 'Y', 'Z']
        )
        x_bin = borders.searchsorted(df_particles['X'].to_numpy())
        y_bin = borders.searchsorted(df_particles['Y'].to_numpy())
        z_bin = borders.searchsorted(df_particles['Z'].to_numpy())

        g = dict((*df_particles.groupby([x_bin, y_bin, z_bin]),))

        g_keys = list(g.keys())

        for cntr, cor in enumerate(g_keys):
            my_tensor[cor[0]-1, cor[1]-1, cor[2]-1] = 1

        for _ in tqdm(range(int(num_samples))):
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

            lattices.append(np.expand_dims(my_tensor[i:i+x_size, j:j+y_size, k:k + z_size], axis=0))
            boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
            cntr += 1
            if cntr % (samples_per_snapshot) == 0:
                num_sample = R.randint(num_frames)
                df_particles = pd.DataFrame(
                blocks[num_sample],
                columns=['X', 'Y', 'Z']
                )

            x_bin = borders.searchsorted(df_particles['X'].to_numpy())
            y_bin = borders.searchsorted(df_particles['Y'].to_numpy())
            z_bin = borders.searchsorted(df_particles['Z'].to_numpy())

            g = dict((*df_particles.groupby([x_bin, y_bin, z_bin]),))

            g_keys = list(g.keys())

            for cntr, cor in enumerate(g_keys):
                my_tensor[cor[0]-1, cor[1]-1, cor[2]-1] = 1
        
        saved_directory = os.path.join('./data', f'{num_boxes}', f'{x_size}_{y_size}_{z_size}')
        mice.folder_checker(saved_directory)
        with h5py.File(os.path.join(saved_directory, 'data.h5'), "w") as hf:
            hf.create_dataset('dataset_1', data=np.array(lattices))
    
    saved_directory = os.path.join('./data', f'{num_boxes}')
    with open(os.path.join(saved_directory, 'explain'), 'w') as f:
        f.write(f'Metadata regarding the data:\n')
        f.write('='*30)
        f.write(f'\n\nnum samples: {num_samples}\nsamples per snapshot: {samples_per_snapshot}')

    return None

if __name__ == '__main__':
    num_samples = 4e4
    samples_per_snapshot = 1e0
    lat_saver(num_samples, samples_per_snapshot)