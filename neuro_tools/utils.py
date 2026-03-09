"""
utils.py — General-purpose statistical and signal-processing helpers.
"""

import numpy as np
from scipy.stats import zscore


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def zscore_timeseries(data, axis=-1):
    """
    Z-score *data* along the time axis.

    Parameters
    ----------
    data : np.ndarray
    axis : int
        Axis corresponding to time. Default -1 (last axis).

    Returns
    -------
    np.ndarray — same shape as *data*
    """
    return zscore(data, axis=axis)


# ---------------------------------------------------------------------------
# Correlation / similarity
# ---------------------------------------------------------------------------

def correlation_matrix(data):
    """
    Compute the Pearson correlation matrix of a (n_features × T) matrix.

    Returns (n_features × n_features) matrix.
    """
    return np.corrcoef(data)


def time_by_time_correlation(data):
    """
    Compute a (T × T) time-by-time correlation matrix.

    Parameters
    ----------
    data : np.ndarray, shape (n_features, T)

    Returns
    -------
    np.ndarray, shape (T, T)
    """
    return np.corrcoef(data.T)


def proportion_variance_explained(A, B):
    """
    Proportion of variance in *A* explained by *B* via MSE.

    Both arrays must have the same shape (n_features, T).

    Parameters
    ----------
    A, B : np.ndarray

    Returns
    -------
    pve : float
    """
    assert A.shape == B.shape, "A and B must have the same shape."
    A = A - np.mean(A)
    B = B - np.mean(B)
    mse = np.mean((A - B) ** 2)
    total_var = np.var(A)
    return 1.0 - (mse / total_var) if total_var > 0 else np.nan


# ---------------------------------------------------------------------------
# Bimodality / diptest
# ---------------------------------------------------------------------------

def run_diptest(data):
    """
    Apply Hartigan's dip test for bimodality to every row of the
    meta-correlation matrix derived from *data*.

    Parameters
    ----------
    data : np.ndarray, shape (n_features, T)

    Returns
    -------
    dips : np.ndarray, shape (T,)
    dip_pvals : np.ndarray, shape (T,)
    """
    try:
        import diptest
    except ImportError as exc:
        raise ImportError(
            "diptest is required. Install with: pip install diptest"
        ) from exc

    TbyT = np.corrcoef(data.T)
    meta = np.corrcoef(TbyT)
    n = len(meta)

    dips, pvals = [], []
    for i in range(n):
        d, p = diptest.diptest(meta[i, :].flatten())
        dips.append(d)
        pvals.append(p)

    return np.asarray(dips), np.asarray(pvals)


# ---------------------------------------------------------------------------
# Boundary / event helpers
# ---------------------------------------------------------------------------

def find_label_boundaries(labels):
    """
    Identify segment boundaries in a 1-D label sequence.

    Parameters
    ----------
    labels : array-like

    Returns
    -------
    boundaries : list of int
        Indices where a new label begins (including 0 and len(labels)).
    region_sizes : list of int
    unique_labels : list
    """
    labels = list(labels)
    boundaries = [0]
    region_sizes = []
    unique_labels = []

    current = labels[0]
    start = 0

    for i, lbl in enumerate(labels[1:], start=1):
        if lbl != current:
            boundaries.append(i)
            region_sizes.append(i - start)
            unique_labels.append(current)
            current = lbl
            start = i

    boundaries.append(len(labels))
    region_sizes.append(len(labels) - start)
    unique_labels.append(current)

    return boundaries, region_sizes, unique_labels


def compute_boundary_overlap(event_indices, state_indices, n):
    """
    Absolute (OA) and relative (OR) boundary-overlap metrics.

    Parameters
    ----------
    event_indices : array-like
        Indices of annotated event boundaries.
    state_indices : array-like
        Indices of neural state boundaries.
    n : int
        Total number of timepoints (for chance correction).

    Returns
    -------
    OA : float — absolute overlap above chance
    OR : float — relative overlap above chance
    """
    event_indices = np.asarray(event_indices)
    state_indices = np.asarray(state_indices)

    num_E = len(event_indices)
    num_S = len(state_indices)
    O = len(np.intersect1d(event_indices, state_indices))
    OE = (num_E * num_S) / n  # expected by chance

    denom_A = num_S - OE
    OA = (O - OE) / denom_A if denom_A != 0 else 0.0

    denom_R = min(num_E, num_S) - OE
    OR = (O - OE) / denom_R if denom_R != 0 else 0.0

    return OA, OR


def shuffle_preserving_intervals(event_indices, rng=None):
    """
    Shuffle event boundaries while preserving inter-event interval distribution.

    Parameters
    ----------
    event_indices : np.ndarray, shape (n_events,)
    rng : np.random.Generator or None

    Returns
    -------
    shuffled : np.ndarray, shape (n_events,)
    """
    if rng is None:
        rng = np.random.default_rng()

    ieis = np.diff(event_indices)
    shuffled_ieis = rng.permutation(ieis)
    shuffled = np.concatenate([[event_indices[0]], np.cumsum(shuffled_ieis)])
    return shuffled
