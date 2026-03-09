"""
dynamics.py — Dynamical-systems analysis of state-space trajectories.

Functions
---------
find_fixed_points   : Detect and classify fixed points of the empirical
                      velocity field (stable/unstable nodes, spirals, saddles).

Internal helpers
----------------
_build_velocity_grid : Interpolate empirical velocities onto a regular grid.
_classify_eigenvalues: Map Jacobian eigenvalues to a fixed-point type string.
"""

import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_velocity_grid(scores, pc_x, pc_y, n_grid, smooth_sigma=2.0):
    """
    Interpolate empirical step-by-step velocities onto a regular (n_grid × n_grid)
    mesh and lightly smooth to reduce TR-level noise.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
    pc_x, pc_y : int
        1-indexed PC axes.
    n_grid : int
        Grid resolution.
    smooth_sigma : float
        Gaussian smoothing sigma (in grid cells). Higher = smoother field.

    Returns
    -------
    xg, yg : np.ndarray, shape (n_grid,)
        Grid axis coordinates.
    U, V : np.ndarray, shape (n_grid, n_grid)
        x- and y-components of the smoothed velocity field.
        Rows index yg, columns index xg.
    valid : np.ndarray of bool, shape (n_grid, n_grid)
        True where griddata could interpolate (i.e. inside the convex hull of data).
    """
    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]
    dx = np.diff(x)
    dy = np.diff(y)
    xm = (x[:-1] + x[1:]) / 2
    ym = (y[:-1] + y[1:]) / 2

    xg = np.linspace(x.min(), x.max(), n_grid)
    yg = np.linspace(y.min(), y.max(), n_grid)
    XXg, YYg = np.meshgrid(xg, yg)

    U_raw = griddata((xm, ym), dx, (XXg, YYg), method="linear", fill_value=np.nan)
    V_raw = griddata((xm, ym), dy, (XXg, YYg), method="linear", fill_value=np.nan)

    valid = np.isfinite(U_raw) & np.isfinite(V_raw)

    # Gaussian-smooth; fill NaN regions with 0 before smoothing, restore after
    U = gaussian_filter(np.where(valid, U_raw, 0.0), sigma=smooth_sigma)
    V = gaussian_filter(np.where(valid, V_raw, 0.0), sigma=smooth_sigma)

    # Zero out cells that had no data (avoid artefacts from zero-fill at edges)
    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)

    return xg, yg, U, V, valid


def _classify_eigenvalues(eigvals):
    """
    Map a pair of Jacobian eigenvalues to a fixed-point type string.

    Type strings
    ------------
    'stable_node'     : both eigenvalues real, negative
    'unstable_node'   : both eigenvalues real, positive
    'saddle'          : eigenvalues real, opposite signs
    'stable_spiral'   : complex conjugate pair, negative real part
    'unstable_spiral' : complex conjugate pair, positive real part
    'center'          : purely imaginary (zero real part)
    """
    r = eigvals.real
    has_imag = np.any(np.abs(eigvals.imag) > 1e-8 * np.abs(eigvals).max())

    if has_imag:
        mean_r = r.mean()
        if mean_r < -1e-10:
            return "stable_spiral"
        elif mean_r > 1e-10:
            return "unstable_spiral"
        else:
            return "center"
    else:
        if np.all(r < 0):
            return "stable_node"
        elif np.all(r > 0):
            return "unstable_node"
        else:
            return "saddle"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_fixed_points(
    scores,
    pc_x=1,
    pc_y=2,
    n_grid=60,
    smooth_sigma=2.0,
    min_speed_pct=8,
    dedup_tol=None,
):
    """
    Detect and classify fixed points of the empirical velocity field.

    Fixed points are located where both velocity components vanish
    (``dx/dt = 0`` **and** ``dy/dt = 0``).  They are found in two steps:

    1. Identify local minima of the speed ``||(U, V)||`` below
       ``min_speed_pct``-th percentile as candidate positions.
    2. Refine each candidate with ``scipy.optimize.fsolve`` and classify
       via the Jacobian eigenvalues.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_components)
        PC scores from :func:`~neuro_tools.decomposition.compute_pca`.
    pc_x, pc_y : int
        1-indexed PC axes (default PC1 vs PC2).
    n_grid : int
        Velocity-field grid resolution.  Higher = more precise detection but
        slower.  60 is a good default for fMRI TR-length series.
    smooth_sigma : float
        Gaussian smoothing of the velocity field (grid cells).
        Increase if the field is too noisy and spurious fixed points appear.
    min_speed_pct : float
        Percentile threshold below which a local speed minimum is considered
        a fixed-point candidate (0–100).  Lower = stricter.
    dedup_tol : float or None
        Two refined solutions closer than this distance are merged.
        Defaults to 2 × grid spacing.

    Returns
    -------
    fixed_points : list of dict
        Each dict contains:

        ``'x'``, ``'y'``
            Position in PC space.
        ``'type'``
            Classification string: ``'stable_node'``, ``'unstable_node'``,
            ``'saddle'``, ``'stable_spiral'``, ``'unstable_spiral'``,
            ``'center'``.
        ``'eigenvalues'``
            Complex ndarray, shape (2,): Jacobian eigenvalues.
        ``'jacobian'``
            2×2 Jacobian matrix at the fixed point.
    """
    xg, yg, U, V, valid = _build_velocity_grid(
        scores, pc_x, pc_y, n_grid, smooth_sigma
    )

    # Replace NaN with large value so local minima don't land outside data
    U_fill = np.where(valid, U, np.nanmax(np.abs(U[valid])) * 10 + 1)
    V_fill = np.where(valid, V, np.nanmax(np.abs(V[valid])) * 10 + 1)
    speed = np.sqrt(U_fill ** 2 + V_fill ** 2)

    # Find local speed minima below threshold
    local_min = speed == minimum_filter(speed, size=max(3, n_grid // 12))
    threshold = np.nanpercentile(speed[valid], min_speed_pct)
    candidates = np.argwhere(local_min & (speed < threshold) & valid)

    if len(candidates) == 0:
        return []

    # Build interpolators for root-finding
    U_interp = RegularGridInterpolator(
        (yg, xg), np.where(valid, U, 0.0), bounds_error=False, fill_value=0.0
    )
    V_interp = RegularGridInterpolator(
        (yg, xg), np.where(valid, V, 0.0), bounds_error=False, fill_value=0.0
    )

    def velocity(pos):
        pt = np.array([[pos[1], pos[0]]])  # (y, x) for RGI
        return [float(U_interp(pt)), float(V_interp(pt))]

    # Jacobian components via numerical gradients
    dUdy, dUdx = np.gradient(np.where(valid, U, 0.0), yg, xg)
    dVdy, dVdx = np.gradient(np.where(valid, V, 0.0), yg, xg)

    dUdx_i = RegularGridInterpolator((yg, xg), dUdx, bounds_error=False, fill_value=0.0)
    dUdy_i = RegularGridInterpolator((yg, xg), dUdy, bounds_error=False, fill_value=0.0)
    dVdx_i = RegularGridInterpolator((yg, xg), dVdx, bounds_error=False, fill_value=0.0)
    dVdy_i = RegularGridInterpolator((yg, xg), dVdy, bounds_error=False, fill_value=0.0)

    if dedup_tol is None:
        dedup_tol = 2.0 * max(xg[1] - xg[0], yg[1] - yg[0])

    fixed_points = []
    seen_positions = []

    for r, c in candidates:
        x0 = [xg[c], yg[r]]
        try:
            sol, _, ier, _ = fsolve(velocity, x0, full_output=True)[:4]
        except Exception:
            continue

        if ier != 1:
            continue  # fsolve did not converge

        # Keep within grid bounds
        if not (xg.min() <= sol[0] <= xg.max() and yg.min() <= sol[1] <= yg.max()):
            continue

        # Deduplicate
        if any(np.linalg.norm(sol - s) < dedup_tol for s in seen_positions):
            continue
        seen_positions.append(sol)

        pt = np.array([[sol[1], sol[0]]])  # (y, x)
        J = np.array([
            [float(dUdx_i(pt)), float(dUdy_i(pt))],
            [float(dVdx_i(pt)), float(dVdy_i(pt))],
        ])
        eigvals = np.linalg.eigvals(J)
        fp_type = _classify_eigenvalues(eigvals)

        fixed_points.append({
            "x": float(sol[0]),
            "y": float(sol[1]),
            "type": fp_type,
            "eigenvalues": eigvals,
            "jacobian": J,
        })

    return fixed_points
