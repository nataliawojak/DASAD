from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree


def kl_divergence(p, q, eps=.01):
    """Estimate KL divergence between two distributions using nearest neighbors.

    Args:
        p (np.ndarray): Samples from distribution P.
        q (np.ndarray): Samples from distribution Q.
        eps (float, optional): Approximation parameter for KDTree search.
            Smaller values give more accurate but slower results. Defaults to 0.01.

    Returns:
        float: Estimated KL divergence between distributions P and Q.

    Raises:
        AssertionError: If input distributions have different dimensions.
    """

    p = np.atleast_2d(p)
    q = np.atleast_2d(q)

    n,d = p.shape
    m,dy = q.shape
    
    assert(d == dy)
    
    
    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(p)
    ytree = KDTree(q)
    
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(p, k=2, eps=eps, p=2)[0][:,1]
    s = ytree.query(p, k=1, eps=eps, p=2)[0]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    log_ = -np.log(r/s)
    log_= log_[np.isfinite(log_)]

    kl_div = log_.sum() * d / n + np.log(m / (n - 1.))
        
    return kl_div



def run_drift_detection(detector, df):
    """Run drift detection sequentially on dataset.

    Args:
        detector: Drift detection object implementing `update()` method.
        df (pd.DataFrame): Input dataset where each row is an observation.

    Returns:
        list[int]: Indices where drift was detected.
    """
    drift_points = []

    for i in range(len(df)):
        x_i = df.iloc[i].values

        drift = detector.update(x_i)
 

        if drift:
            drift_points.append(i)

    return drift_points


def make_json_serializable(value):
    """Convert nested experiment results to JSON-serializable objects.

    The function recursively processes dictionaries, lists, tuples, NumPy
    arrays and NumPy scalar values. Non-finite floating-point values, such as
    NaN and infinity, are converted to None.

    Args:
        value: Object or nested collection to convert to a JSON-serializable
            representation.

    Returns:
        JSON-serializable object containing only dictionaries, lists, strings,
        integers, finite floats, booleans, and None.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_serializable(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value