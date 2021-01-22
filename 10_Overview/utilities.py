import numpy as np
import pandas as pd


def typecheck_array(array, name="array"):
    assert isinstance(
        array, np.ndarray
    ), "{} must be represented as a NumPy array".format(name)

def typecheck_series(dataset, name="dataset"):
    assert isinstance(
        dataset, pd.Series
    ), "{} must be represented by a Pandas Series".format(name)