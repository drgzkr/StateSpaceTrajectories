# StateSpaceTrajectories

Exploratory computational cognitive neuroscience analyses of naturalistic fMRI data (movie watching). Computes PCA on multivariate BOLD signals and visualises brain-state trajectories in low-dimensional PC space.

## Notebooks

| Notebook | Description | Colab |
|---|---|---|
| `StateSpaceTrajectories.ipynb` | Main analysis: whole-brain ROI-averaged PCA and within-ROI voxel-level PCA with 2D/3D trajectory plots | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/StateSpaceTrajectories/blob/main/StateSpaceTrajectories.ipynb) |

> **Private repo note:** clicking the Colab badge will prompt you to authorise GitHub access — this works automatically since you own the repository.

## `neuro_tools` — Personal Toolbox

A small, reusable Python package extracted from recurring patterns across analysis notebooks. Import it in any Colab notebook by copying it from Drive or cloning this repo.

```
neuro_tools/
├── atlas.py         — Schaefer 2018 & Julich-Brain atlas loading / resampling
├── io.py            — ROI extraction, ROI-averaged matrices, NIfTI projection
├── decomposition.py — PCA wrappers (compute_pca, get_pc_trajectory)
├── plotting.py      — Brain surface plots, 2D/3D trajectory plots, scree plots
└── utils.py         — z-scoring, correlation, diptest, boundary overlap helpers
```

### Quick-start (Colab)

```python
# Clone the repo inside Colab
!git clone https://github.com/drgzkr/StateSpaceTrajectories.git
import sys; sys.path.insert(0, 'StateSpaceTrajectories')

from neuro_tools import atlas as nt_atlas
from neuro_tools import io as nt_io
from neuro_tools import decomposition as nt_dec
from neuro_tools import plotting as nt_plot
from neuro_tools import utils as nt_utils
```

### Key functions

| Module | Function | Description |
|---|---|---|
| `atlas` | `load_schaefer_atlas(reference_img, n_rois=400)` | Fetch & resample Schaefer atlas |
| `atlas` | `load_julich_roi_mask(roi_list, reference_img)` | Build mask from Julich-Brain ROI names |
| `io` | `get_roi_data(roi_idx, whole_brain_data, atlas)` | Voxels × time for one ROI |
| `io` | `compute_roi_averaged_matrix(data, atlas, n_rois)` | ROIs × time matrix (z-scored) |
| `io` | `roi_pattern_to_nifti(pattern, atlas)` | Project ROI values back to 3-D NIfTI |
| `decomposition` | `compute_pca(data, n_components=10)` | PCA → scores, components, EVR |
| `plotting` | `plot_trajectories_grid(scores, pairs=[(1,2),(2,3),(1,3)])` | Grid of 2-D trajectory panels |
| `plotting` | `plot_trajectory_3d(scores)` | 3-D trajectory coloured by time |
| `plotting` | `long_plot(tex_l, tex_r, ...)` | 4-panel fsaverage surface plot |
| `plotting` | `nifti_to_surface(nifti_image)` | Project volume → surface textures |
| `plotting` | `plot_explained_variance(evr)` | Scree plot |
| `utils` | `run_diptest(data)` | Bimodality test on meta-correlation matrix |
| `utils` | `compute_boundary_overlap(events, states, n)` | Boundary overlap metrics (OA, OR) |

## Dependencies

```
nilearn
nibabel
scikit-learn
scipy
numpy
matplotlib
tqdm
diptest        # for bimodality testing
siibra         # optional, only for Julich atlas
```
