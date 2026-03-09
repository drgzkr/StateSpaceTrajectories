"""
io.py — ROI data extraction from whole-brain volumetric arrays.
"""

import numpy as np
from scipy.stats import zscore as _zscore


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def get_roi_data(roi_idx, whole_brain_data, atlas_resampled, clean=True):
    """
    Extract voxel-by-time data for a single ROI.

    Parameters
    ----------
    roi_idx : int
        Atlas label index for the desired ROI (1-indexed for Schaefer).
    whole_brain_data : np.ndarray, shape (x, y, z, T)
        4-D whole-brain data array.
    atlas_resampled : Nifti1Image
        Resampled atlas image (same voxel space as *whole_brain_data*).
    clean : bool
        If True, remove all-zero voxels (empty voxels). Default True.

    Returns
    -------
    roi_data : np.ndarray, shape (n_voxels, T)
        Voxel-by-time matrix for the ROI.
    """
    atlas_data = atlas_resampled.get_fdata()
    roi_data = whole_brain_data[atlas_data == roi_idx, :]

    if clean:
        roi_data = roi_data[~np.all(roi_data == 0, axis=1)]

    return roi_data


def get_roi_average(roi_idx, whole_brain_data, atlas_resampled, clean=True):
    """
    Return the mean time-series across voxels in a single ROI.

    Parameters
    ----------
    roi_idx, whole_brain_data, atlas_resampled, clean
        See :func:`get_roi_data`.

    Returns
    -------
    mean_ts : np.ndarray, shape (T,)
    """
    roi_data = get_roi_data(roi_idx, whole_brain_data, atlas_resampled, clean=clean)
    return np.nanmean(roi_data, axis=0)


def compute_roi_averaged_matrix(
    whole_brain_data,
    atlas_resampled,
    n_rois,
    zscore_output=True,
):
    """
    Build a (n_rois × T) matrix of ROI-averaged BOLD time-series.

    ROIs whose mean is NaN or all-zero are replaced with small Gaussian noise
    (preserving array shape while flagging missing data).

    Parameters
    ----------
    whole_brain_data : np.ndarray, shape (x, y, z, T)
    atlas_resampled : Nifti1Image
    n_rois : int
        Total number of ROIs in the atlas.
    zscore_output : bool
        Z-score each ROI time-series across time. Default True.

    Returns
    -------
    roi_matrix : np.ndarray, shape (n_rois, T)
    """
    n_timepoints = whole_brain_data.shape[-1]
    roi_matrix = np.zeros((n_rois, n_timepoints))

    for i in range(1, n_rois + 1):
        mean_ts = get_roi_average(i, whole_brain_data, atlas_resampled)

        if np.isnan(mean_ts[0]) or np.all(mean_ts == 0):
            # Replace missing ROI with negligible noise
            roi_matrix[i - 1, :] = np.random.randn(n_timepoints) * 0.01
        else:
            roi_matrix[i - 1, :] = mean_ts

    if zscore_output:
        roi_matrix = _zscore(roi_matrix, axis=-1)

    return roi_matrix


def get_mask_roi_data(roi_coords, whole_brain_data, zscore_output=False):
    """
    Extract voxel-by-time data using arbitrary coordinate masks
    (e.g. from Julich atlas).

    Parameters
    ----------
    roi_coords : tuple of np.ndarray
        Output of ``np.where(mask > 0)``.
    whole_brain_data : np.ndarray, shape (x, y, z, T)
    zscore_output : bool
        Z-score each voxel time-series across time. Default False.

    Returns
    -------
    roi_data : np.ndarray, shape (n_voxels, T)
    """
    roi_data = whole_brain_data[roi_coords]

    if zscore_output:
        roi_data = _zscore(roi_data, axis=-1)

    return roi_data


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def roi_pattern_to_nifti(pattern, atlas_resampled, saving=False, output_path=""):
    """
    Project a 1-D ROI-level pattern back into volumetric NIfTI space.

    Parameters
    ----------
    pattern : np.ndarray, shape (n_rois,)
        One value per atlas ROI (e.g. PC loadings, t-statistics).
    atlas_resampled : Nifti1Image
        Resampled atlas image.
    saving : bool
        Save to *output_path* if True.
    output_path : str
        File path for saving (only used when *saving* is True).

    Returns
    -------
    new_img : Nifti1Image
    """
    import nibabel as nib

    template_data = atlas_resampled.get_fdata()
    new_data = np.zeros_like(template_data)

    for roi_label in np.unique(template_data):
        if roi_label == 0:
            continue
        idx = int(roi_label) - 1
        if idx < len(pattern):
            new_data[template_data == roi_label] = pattern[idx]

    new_img = nib.Nifti1Image(
        new_data,
        affine=atlas_resampled.affine,
        header=atlas_resampled.header,
    )
    if saving:
        nib.save(new_img, output_path)

    return new_img
