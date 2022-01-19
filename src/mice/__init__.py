import gin
import numpy as np

from mice.neural_net.architectures import (
    Net,
    Model,
    Modely
)

from mice.utils.my_utils import (
    MiceDataset,
    read_data,
    frames,
    sizer,
    mi_model,
    boxes_maker,
    lattices_generator,
    lattice_splitter,
    loss_function,
    train_one_epoch,
    train_one_step,
    valid_one_epoch,
    valid_one_step,
    box_fig,
    box_fig_together,
    box_fig_running,
    entropy_fig,
    entropy_fig_together,
    entropy_fig_running,
    folder_checker,
    sort_func,
    logger,
    print_combinations
)

from mice.main.box_menu import (
    box_runner
)

from mice.main.entropy_menu import (
    entropy_runner
)

from bin.load_data import (
    file_load
)

# gin configurations
gin.external_configurable(np.array)
gin.parse_config_file('./src/mice/config.gin')
