import numpy as np
from a_b import a_b

def test_a_b_returns_numpy_array():
    result = a_b()
    assert isinstance(result, np.ndarray)
