import pytest
import mice
import numpy as np
from collections import namedtuple

lattices_generator_input = namedtuple('input', ['box_frac', 'num_samples', 'samples_per_snapshot', 'R', 'num_frames', 'num_boxes', 'sizes'])
input = lattices_generator_input(box_frac=1, num_samples=2e2, samples_per_snapshot=1e1, R=np.random.RandomState(seed=0), num_frames=1000, num_boxes=4, sizes=(4,4,4))

@pytest.mark.parametrize("test_input,expected", [
    (input, list),
])
def test_box_menu(test_input, expected):
    box_frac, num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes = test_input
    assert type(mice.lattices_generator(num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes)) is expected
