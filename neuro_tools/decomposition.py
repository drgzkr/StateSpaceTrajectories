"""
decomposition.py — PCA and state-space trajectory helpers.
"""

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def compute_pca(data, n_components=10, zscore_input=False):
    """
    Run PCA on a (n_features × n_timepoints) data matrix.

    Convention: rows are features (ROIs / voxels), columns are timepoints.
    PCA is fitted on the *transposed* matrix so that each timepoint is a
    sample and each feature is a variable — giving PC *scores* as a
    (n_timepoints × n_components) trajectory.

    Parameters
    ----------
    data : np.ndarray, shape (n_features, T)
    n_components : int
        Number of PCs to retain. Default 10.
    zscore_input : bool
        Z-score each feature across time before fitting. Default False
        (assume caller has already normalised).

    Returns
    -------
    scores : np.ndarray, shape (T, n_components)
        PC scores (the trajectory of brain states in PC space).
    components : np.ndarray, shape (n_components, n_features)
        PC loadings (spatial maps).
    explained_variance_ratio : np.ndarray, shape (n_components,)
        Fraction of variance explained by each PC.
    pca : sklearn.decomposition.PCA
        Fitted PCA object (for further inspection / transform).
    """
    from scipy.stats import zscore

    X = data.T  # (T, n_features)

    if zscore_input:
        X = zscore(X, axis=0)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)          # (T, n_components)
    components = pca.components_           # (n_components, n_features)
    evr = pca.explained_variance_ratio_

    return scores, components, evr, pca


def get_pc_trajectory(scores, pc_x=1, pc_y=2, pc_z=None):
    """
    Convenience wrapper: extract specific PC axes from a scores matrix.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
        Output of :func:`compute_pca`.
    pc_x, pc_y : int
        1-indexed PC numbers for the two horizontal / vertical axes.
    pc_z : int or None
        If given, also return a third axis (for 3-D plots).

    Returns
    -------
    x, y[, z] : np.ndarray, each shape (T,)
    """
    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]
    if pc_z is not None:
        z = scores[:, pc_z - 1]
        return x, y, z
    return x, y
