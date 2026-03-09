"""
atlas.py — Atlas loading and ROI mask utilities.

Supported atlases
-----------------
- Schaefer 2018 parcellation (via nilearn)
- Julich-Brain (via siibra)
"""

import numpy as np
import nibabel as nib
from nilearn import datasets, image


# ---------------------------------------------------------------------------
# Schaefer atlas
# ---------------------------------------------------------------------------

def load_schaefer_atlas(reference_img, n_rois=400, yeo_networks=17, resolution_mm=2):
    """
    Fetch the Schaefer 2018 atlas and resample it to match *reference_img*.

    Parameters
    ----------
    reference_img : str or Nifti1Image
        Path to, or loaded, reference functional image.
    n_rois : int
        Number of ROIs in the Schaefer parcellation (100 / 200 / 300 / 400 /
        500 / 600 / 800 / 1000). Default 400.
    yeo_networks : int
        Yeo network resolution (7 or 17). Default 17.
    resolution_mm : int
        Atlas resolution in mm (1 or 2). Default 2.

    Returns
    -------
    atlas_resampled : Nifti1Image
        Atlas image in the space of *reference_img*.
    labels : list of str
        ROI label strings (length == n_rois).
    """
    if isinstance(reference_img, str):
        reference_img = image.load_img(reference_img)

    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
    )
    atlas_img = image.load_img(schaefer["maps"])
    labels = schaefer["labels"]

    if isinstance(labels[0], bytes):
        labels = [lbl.decode("utf-8") for lbl in labels]

    atlas_resampled = image.resample_to_img(
        atlas_img, reference_img, interpolation="nearest"
    )
    return atlas_resampled, labels


# ---------------------------------------------------------------------------
# Julich-Brain atlas
# ---------------------------------------------------------------------------

def load_julich_roi_mask(roi_list, reference_img, parcellation="julich 2.9"):
    """
    Build a combined binary mask for a list of Julich-Brain ROI names.

    Requires the *siibra* package.

    Parameters
    ----------
    roi_list : list of str
        ROI names as they appear in the Julich atlas
        (e.g. ``['hOc1 left', 'hOc1 right']``).
    reference_img : str or Nifti1Image
        Reference functional image for resampling.
    parcellation : str
        Siibra parcellation identifier. Default ``'julich 2.9'``.

    Returns
    -------
    roi_mask : np.ndarray, shape (x, y, z)
        Binary mask (0/1) in the space of *reference_img*.
    roi_coords : tuple of np.ndarray
        Tuple of coordinate arrays (output of ``np.where``).
    """
    try:
        import siibra
    except ImportError as exc:
        raise ImportError(
            "siibra is required for Julich atlas support. "
            "Install it with: pip install siibra"
        ) from exc

    if isinstance(reference_img, str):
        reference_img = image.load_img(reference_img)

    roi_coords_mask = np.zeros(reference_img.shape[:3])

    for roi_name in roi_list:
        roi_sii = siibra.get_region(parcellation=parcellation, region=roi_name)
        roi_mask_img = roi_sii.get_regional_map(
            space="mni152", maptype="labelled"
        ).fetch()
        resampled = image.resample_img(
            roi_mask_img,
            target_affine=reference_img.affine,
            target_shape=reference_img.shape[:3],
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        roi_coords_mask[np.where(resampled.get_fdata() > 0)] += 1

    roi_coords = np.where(roi_coords_mask > 0)
    return (roi_coords_mask > 0).astype(int), roi_coords
