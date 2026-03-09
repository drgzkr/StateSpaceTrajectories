"""
neuro_tools — personal toolbox for cognitive neuroscience analyses.

Modules
-------
atlas       : Atlas loading, resampling, and ROI mask creation
io          : ROI data extraction from whole-brain arrays
decomposition : PCA / dimensionality reduction helpers
plotting    : Brain surface plots and trajectory visualisations
utils       : General-purpose helpers (z-scoring, correlation, etc.)
"""

from . import atlas, io, decomposition, plotting, utils
