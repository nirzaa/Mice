import numpy as np
import h5py
import os
import gin
import sys
my_path = os.path.join('./')
sys.path.append(my_path)
import mice

@gin.configurable
def file_load(number_lines):
    '''
    This function will take the dump.0 file that we will get by running the simulation,
    and will generate an .h5 file for future processing of the data.
    '''
    path_data = os.path.join('./', 'data')

    with open(os.path.join(path_data, 'dump.0')) as f:
        lines = [next(f) for x in range(number_lines)]
    f.close()

    mod265 = np.arange(len(lines)) % 265
    blocks = np.array(lines)[(mod265 <= 264) & (mod265 >= 9)]
    blocks = blocks[:-1]
    blocks = np.array([np.array(x.split()[2:], dtype=float) for x in blocks])
    leny = len(blocks)
    leny_mod = leny // 256
    leny_mod = leny_mod * 256
    blocks = blocks[:leny_mod].reshape((-1, 256, 3))
    blocks = blocks.astype(float)

    hf = h5py.File(os.path.join(path_data, 'data.h5'), 'w')
    hf.create_dataset('dataset_1', data=blocks)
    hf.close()

    # sanity checks:

    my_hf = h5py.File(os.path.join(path_data,'data.h5'), 'r')
    n1 = my_hf.get('dataset_1')
    blocks = np.array(n1)
    my_hf.close()

    print(blocks.shape)

    print(blocks[0, 0])
    print(blocks[0, 255])

    print(blocks[1, 0])
    print(blocks[1, 255])
    
    return None

if __name__ == '__main__':
    '''
    This module will take the dump.0 file that we will get by running the simulation,
    and will generate an .h5 file for future processing of the data.
    '''
    mice.file_load()
    