from abc import ABC, abstractmethod
import logging
import os

from typing import List, Tuple, Union, Optional

import numpy as np
import glob
import partitura as pt
import warnings

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

HOME_DIR = "."

if IN_COLAB:
    HOME_DIR = "/content/miws2024/expectation"

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


QUANTIZED_DURATIONS = np.array(
    [
        2,
        1,
        0.75,
        0.5,
        0.25,
    ]
)
QUANTIZED_DURATIONS.sort()

def load_data(min_seq_length: int = 10) -> List[np.ndarray]:
    # load data
    files = glob.glob(os.path.join(HOME_DIR, "data", "*.mid"))
    files.sort()
    sequences = []
    for fn in files:
        seq = pt.load_performance_midi(fn)[0]
        if len(seq.notes) > min_seq_length:
            sequences.append(seq.note_array())
    return sequences


def find_nearest(array: np.ndarray, value: float) -> np.ndarray:
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"), 0, len(array) - 1)
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx


def get_indices_cartesian_product(
    elem: Union[List[Tuple], np.ndarray],
    cartesian_product: np.ndarray,
) -> np.ndarray:
    # Convert elem to a numpy array for vectorized comparison
    elem_array = np.array(elem)

    # Find indices where all elements match
    indices = np.where((cartesian_product[:, None] == elem_array).all(-1))[1]

    return indices.astype(np.int32)
