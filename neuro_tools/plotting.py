"""
plotting.py — Visualisation utilities for neuroimaging and state-space analyses.

Functions
---------
Brain surface
    nifti_to_surface      : Project volumetric NIfTI onto fsaverage surface
    long_plot             : 4-panel lateral/medial surface plot

State-space trajectories
    plot_trajectory_2d    : 2-D PC-space trajectory
    plot_trajectories_grid: Grid of 2-D trajectory panels (PC1v2, PC2v3, …)
    plot_trajectory_3d    : Interactive 3-D trajectory with matplotlib
    plot_density_2d       : KDE dwell-time density map
    plot_flow_field       : KDE density + binned average-velocity arrows
    plot_streamlines      : KDE density + continuous streamlines (topology)
    plot_landscape_3d     : 3-D surface of KDE density (energy landscape)
    plot_phase_portrait   : Full phase portrait — density, streamlines,
                            nullclines, and classified fixed points

Timeseries
    plot_data_with_template : Heatmap + 1-D timeseries panel
    plot_explained_variance : Scree plot

Colormaps / helpers
    add_colorbar          : Add a standalone colorbar to a figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Default colour palette
# ---------------------------------------------------------------------------
CSCALE = [
    "#B78D64", "#A4B161", "#3F6871",
    "#764177", "#754C24", "#657123",
    "#4A184C", "#183F48",
]


# ---------------------------------------------------------------------------
# Brain surface helpers
# ---------------------------------------------------------------------------

def nifti_to_surface(nifti_image, zero_threshold=5e-4):
    """
    Project a 3-D NIfTI image onto the fsaverage pial surface.

    Parameters
    ----------
    nifti_image : Nifti1Image
        3-D volumetric image (single volume).
    zero_threshold : float
        Values whose absolute value is below this threshold are set to NaN
        (rendered as transparent). Default 5e-4.

    Returns
    -------
    texture_left, texture_right : np.ndarray
        Per-vertex surface textures for left / right hemispheres.
    """
    from nilearn import datasets, surface

    fsaverage = datasets.fetch_surf_fsaverage()

    texture_left = surface.vol_to_surf(
        nifti_image, fsaverage.pial_left,
        interpolation="nearest_most_frequent", radius=3.0, n_samples=20,
    )
    texture_right = surface.vol_to_surf(
        nifti_image, fsaverage.pial_right,
        interpolation="nearest_most_frequent", radius=3.0, n_samples=20,
    )

    # Mask near-zero values → transparent
    for tex in (texture_left, texture_right):
        tex[np.abs(tex) < zero_threshold] = np.nan

    return texture_left, texture_right


def long_plot(
    texture_left,
    texture_right,
    title="Title",
    saving=False,
    save_path=None,
    min_val=-1,
    max_val=1,
    views=("lateral", "medial"),
    cmap="Spectral_r",
    norm=None,
    show=True,
):
    """
    4-panel brain surface plot (lateral & medial × left & right).

    Parameters
    ----------
    texture_left, texture_right : np.ndarray
        Per-vertex surface textures (from :func:`nifti_to_surface`).
    title : str
    saving : bool
        Save figure if True.
    save_path : str or None
        Full file path for saving. Required when *saving* is True.
    min_val, max_val : float
        Colorbar range.
    views : sequence of 2 str
        Two surface views to display, e.g. ``('lateral', 'medial')``.
    cmap : str
    norm : matplotlib Normalize or None
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from nilearn import datasets, plotting

    fsaverage = datasets.fetch_surf_fsaverage()

    fig, axes = plt.subplots(
        1, 4, figsize=(15, 3), subplot_kw={"projection": "3d"}
    )
    fig.suptitle(title, fontsize=12, x=0.45)

    panels = [
        (views[0], "left"),
        (views[0], "right"),
        (views[1], "left"),
        (views[1], "right"),
    ]

    if norm is None:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)

    for ax, (view, hemi) in zip(axes, panels):
        texture = texture_left if hemi == "left" else texture_right
        surf   = fsaverage.infl_left  if hemi == "left" else fsaverage.infl_right
        bg     = fsaverage.sulc_left  if hemi == "left" else fsaverage.sulc_right

        plotting.plot_surf_stat_map(
            surf, texture,
            hemi=hemi, view=view,
            colorbar=False,
            bg_map=bg,
            cmap=cmap,
            axes=ax,
            vmax=max_val, vmin=min_val,
            bg_on_data=True,
        )

    fig.subplots_adjust(top=1, right=0.85, wspace=0, hspace=0.00002,
                        left=0.05, bottom=0)

    cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([min_val, max_val])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Metric", rotation=270, labelpad=15)

    if saving and save_path:
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# State-space trajectory plots
# ---------------------------------------------------------------------------

def _make_colormap_line(x, y, color_values=None, cmap="viridis", lw=1.5, ax=None):
    """
    Draw a line whose colour encodes *color_values* (e.g. time).
    Returns the LineCollection so the caller can add a colorbar.
    """
    if ax is None:
        ax = plt.gca()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if color_values is None:
        color_values = np.linspace(0, 1, len(x))

    norm = plt.Normalize(color_values.min(), color_values.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(color_values)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_trajectory_2d(
    scores,
    pc_x=1,
    pc_y=2,
    color_values=None,
    cmap="viridis",
    title=None,
    xlabel=None,
    ylabel=None,
    lw=1.5,
    mark_start=True,
    ax=None,
    show_colorbar=True,
    colorbar_label="Time (TR)",
):
    """
    Plot a 2-D state-space trajectory coloured by *color_values*.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
        PC scores from :func:`~neuro_tools.decomposition.compute_pca`.
    pc_x, pc_y : int
        1-indexed PC axes to plot.
    color_values : np.ndarray, shape (T,) or None
        Values used to colour the trajectory (default: time index).
    cmap : str
        Matplotlib colormap name.
    title, xlabel, ylabel : str or None
    lw : float
        Line width.
    mark_start : bool
        Add a star marker at the trajectory start. Default True.
    ax : matplotlib Axes or None
        Draw into existing axes if provided.
    show_colorbar : bool
    colorbar_label : str

    Returns
    -------
    ax : matplotlib Axes
    lc : LineCollection (for external colorbar control)
    """
    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]

    if color_values is None:
        color_values = np.arange(len(x), dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    lc = _make_colormap_line(x, y, color_values=color_values, cmap=cmap, lw=lw, ax=ax)

    if mark_start:
        ax.plot(x[0], y[0], "*", ms=10, color="white",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_xlabel(xlabel or f"PC {pc_x}")
    ax.set_ylabel(ylabel or f"PC {pc_y}")
    ax.set_title(title or f"PC {pc_x} vs PC {pc_y}")
    ax.spines[["top", "right"]].set_visible(False)

    if show_colorbar:
        plt.colorbar(lc, ax=ax, label=colorbar_label, shrink=0.8)

    return ax, lc


def plot_trajectories_grid(
    scores,
    pairs=((1, 2), (2, 3), (1, 3)),
    color_values=None,
    cmap="viridis",
    suptitle=None,
    figsize=None,
    lw=1.5,
):
    """
    Grid of 2-D trajectory panels for multiple PC pairs.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pairs : sequence of (int, int)
        PC pairs to plot. Default: (1,2), (2,3), (1,3).
    color_values : np.ndarray, shape (T,) or None
    cmap : str
    suptitle : str or None
    figsize : tuple or None
    lw : float

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    """
    n = len(pairs)
    if figsize is None:
        figsize = (4.5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    if color_values is None:
        color_values = np.arange(scores.shape[0], dtype=float)

    for ax, (pc_x, pc_y) in zip(axes, pairs):
        show_cb = (ax is axes[-1])
        plot_trajectory_2d(
            scores, pc_x=pc_x, pc_y=pc_y,
            color_values=color_values, cmap=cmap,
            lw=lw, ax=ax, show_colorbar=show_cb,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)

    fig.tight_layout()
    return fig, np.array(axes)


def plot_trajectory_3d(
    scores,
    pc_x=1,
    pc_y=2,
    pc_z=3,
    color_values=None,
    cmap="viridis",
    title="State-Space Trajectory (3D)",
    lw=1.0,
    figsize=(7, 6),
    elev=25,
    azim=45,
):
    """
    3-D state-space trajectory coloured by *color_values*.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y, pc_z : int
        1-indexed PC axes.
    color_values : np.ndarray, shape (T,) or None
    cmap, title, lw, figsize : standard
    elev, azim : float
        Initial 3-D viewing angles.

    Returns
    -------
    fig, ax : Figure and Axes3D
    """
    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]
    z = scores[:, pc_z - 1]

    if color_values is None:
        color_values = np.arange(len(x), dtype=float)

    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=color_values.min(), vmax=color_values.max())

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for i in range(len(x) - 1):
        c = cmap_obj(norm(color_values[i]))
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=c, lw=lw)

    # Start marker
    ax.scatter(x[0], y[0], z[0], s=60, color="white",
               edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_zlabel(f"PC {pc_z}")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time (TR)", shrink=0.6, pad=0.1)

    return fig, ax


def plot_density_2d(
    scores,
    pc_x=1,
    pc_y=2,
    bandwidth=None,
    levels=8,
    cmap="YlOrRd",
    contour_color="white",
    title=None,
    ax=None,
    figsize=(5, 4),
    resolution=150,
):
    """
    KDE dwell-time density map in 2-D PC space.

    Shows where the brain spends the most time rather than plotting every
    step of the raw trajectory.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y : int
        1-indexed PC axes to plot.
    bandwidth : float or str or None
        Passed to ``scipy.stats.gaussian_kde`` as *bw_method*.
        None uses Scott's rule.
    levels : int
        Number of iso-density contour lines overlaid on the map.
    cmap : str
    contour_color : str
    title : str or None
    ax : Axes or None
    figsize : tuple
    resolution : int
        Grid resolution for KDE evaluation.

    Returns
    -------
    ax : Axes
    """
    from scipy.stats import gaussian_kde

    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]

    xgrid = np.linspace(x.min(), x.max(), resolution)
    ygrid = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xgrid, ygrid)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(xx, yy, density, cmap=cmap, shading="gouraud")
    if levels > 0:
        ax.contour(xx, yy, density, levels=levels,
                   colors=contour_color, linewidths=0.6, alpha=0.55)

    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_title(title or f"Dwell-time density — PC {pc_x} vs PC {pc_y}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.colorbar(im, ax=ax, label="Density (dwell time)", shrink=0.8)

    return ax


def plot_flow_field(
    scores,
    pc_x=1,
    pc_y=2,
    n_bins=15,
    min_count=2,
    bandwidth=None,
    cmap="YlOrRd",
    arrow_color="white",
    arrow_scale=None,
    title=None,
    ax=None,
    figsize=(5, 4),
    resolution=150,
):
    """
    KDE density background with binned average-velocity arrows.

    Combines dwell-time information (background colour) with the dominant
    direction of traversal (arrows) so that the most-used paths stand out.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y : int
        1-indexed PC axes.
    n_bins : int
        Number of bins along each axis for the velocity grid.
    min_count : int
        Minimum number of timepoints in a bin to draw an arrow.
        Suppresses arrows in rarely visited regions.
    bandwidth : float or str or None
        KDE bandwidth (Scott's rule if None).
    cmap : str
    arrow_color : str
    arrow_scale : float or None
        Passed to ``ax.quiver`` *scale*. Smaller = longer arrows.
        Defaults to ``n_bins * 2``.
    title : str or None
    ax : Axes or None
    figsize : tuple
    resolution : int
        Grid resolution for KDE background.

    Returns
    -------
    ax : Axes
    """
    from scipy.stats import gaussian_kde

    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]

    # --- KDE density background ---
    xgrid = np.linspace(x.min(), x.max(), resolution)
    ygrid = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xgrid, ygrid)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(xx, yy, density, cmap=cmap, shading="gouraud")

    # --- Binned velocity field ---
    dx = np.diff(x)
    dy = np.diff(y)
    xm = (x[:-1] + x[1:]) / 2   # midpoint positions
    ym = (y[:-1] + y[1:]) / 2

    xedges = np.linspace(x.min(), x.max(), n_bins + 1)
    yedges = np.linspace(y.min(), y.max(), n_bins + 1)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2

    xi = np.clip(np.digitize(xm, xedges) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(ym, yedges) - 1, 0, n_bins - 1)

    counts = np.zeros((n_bins, n_bins))
    su = np.zeros((n_bins, n_bins))
    sv = np.zeros((n_bins, n_bins))
    for dxi, dyi, xi_, yi_ in zip(dx, dy, xi, yi):
        counts[yi_, xi_] += 1
        su[yi_, xi_] += dxi
        sv[yi_, xi_] += dyi

    valid = counts >= min_count
    u = np.where(valid, su / np.where(counts > 0, counts, 1), np.nan)
    v = np.where(valid, sv / np.where(counts > 0, counts, 1), np.nan)

    # Normalise to unit length so arrow size reflects direction only
    mag = np.sqrt(u ** 2 + v ** 2)
    with np.errstate(invalid="ignore"):
        u_n = u / mag
        v_n = v / mag

    XXc, YYc = np.meshgrid(xc, yc)
    scale = arrow_scale if arrow_scale is not None else n_bins * 2
    ax.quiver(
        XXc[valid], YYc[valid], u_n[valid], v_n[valid],
        color=arrow_color, alpha=0.75,
        scale=scale, width=0.004, headwidth=4,
    )

    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_title(title or f"State-space flow — PC {pc_x} vs PC {pc_y}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.colorbar(im, ax=ax, label="Density (dwell time)", shrink=0.8)

    return ax


def plot_streamlines(
    scores,
    pc_x=1,
    pc_y=2,
    n_grid=30,
    bandwidth=None,
    stream_density=1.5,
    lw=1.2,
    cmap="YlOrRd",
    stream_color="white",
    title=None,
    ax=None,
    figsize=(5, 4),
    resolution=150,
):
    """
    KDE density background with continuous streamlines.

    Streamlines follow the average velocity field interpolated onto a regular
    grid, giving a cleaner topographic picture than discrete arrows.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y : int
        1-indexed PC axes.
    n_grid : int
        Resolution of the velocity grid used for streamline integration.
        Higher = more detailed but slower.
    bandwidth : float or str or None
        KDE bandwidth (Scott's rule if None).
    stream_density : float
        Controls how closely packed the streamlines are (passed to
        ``ax.streamplot``).
    lw : float
        Streamline line width.
    cmap : str
    stream_color : str
    title : str or None
    ax : Axes or None
    figsize : tuple
    resolution : int
        Grid resolution for KDE background.

    Returns
    -------
    ax : Axes
    """
    from scipy.stats import gaussian_kde
    from scipy.interpolate import griddata

    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]

    # --- KDE density background ---
    xgrid = np.linspace(x.min(), x.max(), resolution)
    ygrid = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xgrid, ygrid)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.pcolormesh(xx, yy, density, cmap=cmap, shading="gouraud")

    # --- Velocity field interpolated onto regular grid ---
    dx = np.diff(x)
    dy = np.diff(y)
    xm = (x[:-1] + x[1:]) / 2
    ym = (y[:-1] + y[1:]) / 2

    xg = np.linspace(x.min(), x.max(), n_grid)
    yg = np.linspace(y.min(), y.max(), n_grid)
    XXg, YYg = np.meshgrid(xg, yg)

    U = griddata((xm, ym), dx, (XXg, YYg), method="linear", fill_value=0.0)
    V = griddata((xm, ym), dy, (XXg, YYg), method="linear", fill_value=0.0)

    # Scale line width by local speed so fast transitions are thicker
    speed = np.sqrt(U ** 2 + V ** 2)
    max_speed = speed.max() if speed.max() > 0 else 1.0
    lw_scaled = lw + 1.5 * speed / max_speed

    ax.streamplot(
        xg, yg, U, V,
        density=stream_density,
        color=stream_color,
        linewidth=lw_scaled,
        arrowsize=1.0,
        arrowstyle="->",
    )

    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_title(title or f"State-space streamlines — PC {pc_x} vs PC {pc_y}")
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_landscape_3d(
    scores,
    pc_x=1,
    pc_y=2,
    bandwidth=None,
    cmap="YlOrRd",
    resolution=80,
    title=None,
    figsize=(6, 5),
    elev=35,
    azim=45,
    mark_peaks=True,
    n_peaks=3,
    peak_size=60,
):
    """
    3-D surface of KDE dwell-time density — the state-space energy landscape.

    Height encodes how often the brain visits each region of PC space.
    Local maxima (attractor states) are optionally marked with red dots.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y : int
        1-indexed PC axes.
    bandwidth : float or str or None
        KDE bandwidth (Scott's rule if None).
    cmap : str
    resolution : int
        Grid resolution (higher = smoother surface, slower).
    title : str or None
    figsize : tuple
    elev, azim : float
        Initial 3-D viewing angles.
    mark_peaks : bool
        Scatter red markers at the top *n_peaks* local density maxima.
    n_peaks : int
    peak_size : float
        Marker size for peak dots.

    Returns
    -------
    fig, ax : Figure and Axes3D
    """
    from scipy.stats import gaussian_kde
    from scipy.ndimage import maximum_filter

    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]

    xgrid = np.linspace(x.min(), x.max(), resolution)
    ygrid = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xgrid, ygrid)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        xx, yy, density,
        cmap=cmap, alpha=0.90,
        linewidth=0, antialiased=True,
        rcount=resolution, ccount=resolution,
    )

    if mark_peaks:
        neighbourhood = max(3, resolution // 10)
        local_max = density == maximum_filter(density, size=neighbourhood)
        peak_idx = np.argwhere(local_max)
        peak_vals = density[local_max]
        top = np.argsort(peak_vals)[::-1][:n_peaks]
        for i in top:
            r, c = peak_idx[i]
            ax.scatter(
                xgrid[c], ygrid[r], density[r, c],
                color="crimson", s=peak_size, zorder=6, depthshade=False,
            )

    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_zlabel("Density")
    ax.set_title(title or f"State-space landscape — PC {pc_x} vs PC {pc_y}")
    ax.view_init(elev=elev, azim=azim)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=density.min(), vmax=density.max()),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Dwell-time density", shrink=0.55, pad=0.1)

    return fig, ax


# Fixed-point marker styles
_FP_STYLES = {
    "stable_node":     dict(marker="o", color="#00E676", ms=11, label="Stable node"),
    "unstable_node":   dict(marker="o", color="#FF1744", ms=11, label="Unstable node"),
    "stable_spiral":   dict(marker="*", color="#00E676", ms=15, label="Stable spiral"),
    "unstable_spiral": dict(marker="*", color="#FF1744", ms=15, label="Unstable spiral"),
    "saddle":          dict(marker="D", color="#FF9100", ms=10, label="Saddle point"),
    "center":          dict(marker="s", color="#00B0FF", ms=10, label="Center"),
}


def plot_phase_portrait(
    scores,
    pc_x=1,
    pc_y=2,
    fixed_points=None,
    show_nullclines=True,
    n_grid=35,
    bandwidth=None,
    stream_density=1.3,
    cmap="YlOrRd",
    title=None,
    ax=None,
    figsize=(6, 5),
    resolution=150,
):
    """
    Full phase portrait: KDE density background, streamlines, nullclines,
    and classified fixed points.

    Combines all topographic elements into one canonical dynamical-systems
    figure.  Fixed points are coloured and shaped by type (stable node,
    unstable node, saddle, stable/unstable spiral, center).

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y : int
        1-indexed PC axes.
    fixed_points : list of dict or None
        Output of :func:`~neuro_tools.dynamics.find_fixed_points`.
        If None, fixed points are not drawn.
    show_nullclines : bool
        Overlay ``dx/dt = 0`` (cyan dashed) and ``dy/dt = 0`` (gold dashed)
        nullclines.  Their intersections are the fixed points.
    n_grid : int
        Grid resolution for velocity field and nullclines.
    bandwidth : float or str or None
        KDE bandwidth for the density background.
    stream_density : float
        Streamline density (passed to ``ax.streamplot``).
    cmap : str
    title : str or None
    ax : Axes or None
    figsize : tuple
    resolution : int
        KDE background grid resolution.

    Returns
    -------
    ax : Axes
    """
    from scipy.interpolate import griddata
    from scipy.stats import gaussian_kde
    from matplotlib.lines import Line2D

    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]

    # --- KDE density background ---
    xgrid = np.linspace(x.min(), x.max(), resolution)
    ygrid = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xgrid, ygrid)
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.pcolormesh(xx, yy, density, cmap=cmap, shading="gouraud")

    # --- Velocity field + streamlines ---
    dx = np.diff(x)
    dy = np.diff(y)
    xm = (x[:-1] + x[1:]) / 2
    ym = (y[:-1] + y[1:]) / 2

    xg = np.linspace(x.min(), x.max(), n_grid)
    yg = np.linspace(y.min(), y.max(), n_grid)
    XXg, YYg = np.meshgrid(xg, yg)

    U = griddata((xm, ym), dx, (XXg, YYg), method="linear", fill_value=0.0)
    V = griddata((xm, ym), dy, (XXg, YYg), method="linear", fill_value=0.0)

    speed = np.sqrt(U ** 2 + V ** 2)
    lw = 0.8 + 1.5 * speed / (speed.max() or 1.0)

    ax.streamplot(
        xg, yg, U, V,
        density=stream_density, color="white",
        linewidth=lw, arrowsize=0.9, arrowstyle="->",
    )

    # --- Nullclines ---
    legend_handles = []
    if show_nullclines:
        cs_u = ax.contour(XXg, YYg, U, levels=[0],
                          colors="#00CFFF", linewidths=1.8,
                          linestyles="--", alpha=0.9)
        cs_v = ax.contour(XXg, YYg, V, levels=[0],
                          colors="#FFD700", linewidths=1.8,
                          linestyles="--", alpha=0.9)
        legend_handles += [
            Line2D([0], [0], color="#00CFFF", lw=1.8, ls="--",
                   label=r"$\dot{x}=0$ nullcline"),
            Line2D([0], [0], color="#FFD700", lw=1.8, ls="--",
                   label=r"$\dot{y}=0$ nullcline"),
        ]

    # --- Fixed points ---
    if fixed_points:
        seen_types = set()
        for fp in fixed_points:
            style = _FP_STYLES.get(
                fp["type"],
                dict(marker="x", color="white", ms=10, label=fp["type"]),
            )
            label = style["label"] if fp["type"] not in seen_types else None
            h = ax.plot(
                fp["x"], fp["y"],
                marker=style["marker"], color=style["color"],
                ms=style["ms"], mec="black", mew=0.8,
                ls="none", zorder=10, label=label,
            )
            if label:
                legend_handles.append(h[0])
            seen_types.add(fp["type"])

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=8, framealpha=0.35,
            loc="upper right", frameon=True,
        )

    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_title(title or f"Phase portrait — PC {pc_x} vs PC {pc_y}")
    ax.spines[["top", "right"]].set_visible(False)

    return ax


# ---------------------------------------------------------------------------
# Timeseries / explained-variance helpers
# ---------------------------------------------------------------------------

def plot_explained_variance(evr, n_show=None, ax=None, title="Scree plot"):
    """
    Scree plot of explained variance ratio.

    Parameters
    ----------
    evr : np.ndarray
        ``explained_variance_ratio_`` from PCA.
    n_show : int or None
        Number of components to display. Defaults to all.
    ax : Axes or None
    title : str

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))

    if n_show is None:
        n_show = len(evr)

    pcs = np.arange(1, n_show + 1)
    ax.bar(pcs, evr[:n_show] * 100, color=CSCALE[2], edgecolor="white", lw=0.5)
    ax.plot(pcs, np.cumsum(evr[:n_show]) * 100, "o--",
            color=CSCALE[0], lw=1.5, ms=4, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_data_with_template(
    title,
    roi_data,
    timeseries,
    imvmin=-1,
    imvmax=1,
    start=None,
    until=None,
    figsize=None,
    scale=15,
    y_label="ROIs / Voxels",
    ts_label="Projection",
    ts_color=None,
    cscale=None,
):
    """
    Two-panel figure: ROI/voxel heatmap (top) + 1-D time-series (bottom).

    Parameters
    ----------
    title : str
    roi_data : np.ndarray, shape (n_features, T)
    timeseries : np.ndarray, shape (T,)
        1-D timeseries to display below the heatmap.
    imvmin, imvmax : float
        Heatmap colour range.
    start, until : int or None
        Slice the time axis.
    figsize : tuple or None
    scale : float
        Figure scaling factor.
    y_label, ts_label : str
    ts_color : str or None
    cscale : list or None
        Custom colour palette.

    Returns
    -------
    fig : Figure
    """
    if cscale is None:
        cscale = CSCALE
    if ts_color is None:
        ts_color = cscale[2]
    if start is None:
        start = 0
    if until is None:
        until = timeseries.shape[0]
    if figsize is None:
        figsize = (scale * 2, scale * 1)

    fig, axs = plt.subplots(
        2, 1, figsize=figsize,
        dpi=max(1, int(500 / scale)),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title)

    aspect = "equal" if roi_data.shape[0] == roi_data.shape[1] else "auto"
    axs[0].imshow(
        roi_data[:, start:until], aspect=aspect,
        alpha=0.95, cmap="Spectral_r", interpolation="none",
        vmin=imvmin, vmax=imvmax,
    )
    axs[0].tick_params(top=False, bottom=True, labelbottom=True, labeltop=False)
    axs[0].spines[["top", "right", "left", "bottom"]].set_visible(False)
    axs[0].set_yticks([])
    axs[0].set_ylabel(y_label)

    axs[1].axhline(0, color="k", alpha=0.25)
    axs[1].plot(timeseries[start:until], label=ts_label, c=ts_color)
    axs[1].set_xticks([])
    axs[1].yaxis.set_ticks_position("right")
    axs[1].set_xlabel("Timepoints")
    axs[1].spines[["top", "right", "left", "bottom"]].set_visible(False)
    axs[1].tick_params(axis="y", labelsize=6)

    pos = axs[1].get_position()
    axs[1].set_position([pos.x0, pos.y0 - 0.075, pos.width, pos.height])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Colorbar helper
# ---------------------------------------------------------------------------

def add_colorbar(
    fig, ax, cmap,
    orientation="horizontal",
    bar_thickness=None,
    label_fontsize=10,
    show_ticks=True,
    tick_fontsize=8,
    tick_spacing=0.25,
    vmin=0, vmax=1,
    label=None,
    labelpad=5,
):
    """
    Add a customisable standalone colorbar to a figure.

    Parameters
    ----------
    fig : Figure
    ax : Axes
        Reference axes for positioning.
    cmap : str or Colormap
    orientation : {'horizontal', 'vertical'}
    bar_thickness : float or None
    label_fontsize, tick_fontsize : float
    tick_spacing : float
    vmin, vmax : float
    label : str or None
    labelpad : float

    Returns
    -------
    cbar : matplotlib Colorbar
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation=orientation,
                        shrink=bar_thickness or 0.8)
    if label:
        cbar.set_label(label, fontsize=label_fontsize, labelpad=labelpad)
    if show_ticks:
        ticks = np.arange(vmin, vmax + tick_spacing / 2, tick_spacing)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    return cbar
