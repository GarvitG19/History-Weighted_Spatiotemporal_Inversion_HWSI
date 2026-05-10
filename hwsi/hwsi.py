"""
hwsi.py
-------
History-Weighted Spatiotemporal Inversion (HWSI) and three benchmark
time-lapse ERT inversion strategies.

Implemented methods
-------------------
HWSI_Inversion  : main class — cell-adaptive temporal regularisation
run_independent : independent single-epoch inversion (Daily et al., 1992)
run_difference  : difference inversion (LaBrecque & Yang, 2001)
run_4d          : 4D L2-coupled inversion (Kim et al., 2009)

Utility helpers
---------------
interp_to_grid  : interpolate mesh-based model onto a regular grid
build_valid_mask: construct a spatial mask that excludes poorly constrained
                  far-corner nodes from error evaluation
calc_errors     : compute RMSE, normalised RMSE and MAE for any set of results

References
----------
Daily, W., Ramirez, A., LaBrecque, D., Nitao, J., 1992. Electrical resistivity
    tomography of vadose water movement. Water Resources Research 28, 1429-1442.
Kim, J.H., Yi, M.J., Park, S.G., Kim, J.G., 2009. 4-D inversion of DC
    resistivity monitoring data acquired over a dynamically changing earth
    model. Journal of Applied Geophysics 68, 522-532.
LaBrecque, D.J., Yang, X., 2001. Difference inversion of ERT data: a fast
    inversion method for 3-D in situ monitoring. Journal of Environmental and
    Engineering Geophysics 6, 83-89.
Rücker, C., Günther, T., Wagner, F.M., 2017. pyGIMLi: An open-source library
    for modelling and inversion in geophysics. Computers & Geosciences 109,
    106-123.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
import pygimli.meshtools as mt
from pygimli.physics import ert


# ---------------------------------------------------------------------------
# Convergence control constants
# ---------------------------------------------------------------------------
# Convergence is declared when the mean update norm stops improving
# meaningfully over a sliding window of CONV_WINDOW consecutive iterations,
# provided at least MIN_ITERATIONS have been completed.

REL_CONV_TOL = 0.03    # maximum relative improvement still accepted as plateau
CV_TOL       = 0.15    # maximum coefficient of variation within the window
CONV_WINDOW  = 3       # number of iterations in the plateau check window
MIN_ITERS    = 4       # earliest iteration at which convergence can be declared


# ---------------------------------------------------------------------------
# HWSI_Inversion
# ---------------------------------------------------------------------------

class HWSI_Inversion:
    """
    History-Weighted Spatiotemporal Inversion (HWSI).

    Each mesh cell is assigned its own temporal blending coefficient derived
    from the two most recently accepted model updates.  The coefficient is
    recomputed at every Gauss-Seidel sweep, so the weight field co-evolves
    with the inversion solution throughout the solve.

    Weight computation (t >= 2)
    ---------------------------
    raw weight  :  w_raw[j]  = 1 / (eps + |m[t-1,j] - m[t-2,j]|)
    normalised  :  w_norm[j] = w_raw[j] / median(w_raw)
    clipped     :  w[j]      = min(w_norm[j], P95(w_norm))

    Per-cell blending coefficient
    -----------------------------
    alpha[j] = min( lambda_t * w[j] / (lambda_s + lambda_t * w[j]),  alpha_max )

    Blended estimate (Gauss-Seidel, current pass used as prior)
    ------------------------------------------------------------
    m_blend[j] = (1 - alpha[j]) * m_spatial[j]  +  alpha[j] * m_prev[j]

    where m_spatial is the independently inverted model at the current epoch
    and m_prev is the freshest available estimate of the preceding epoch
    within the ongoing sweep.

    Parameters
    ----------
    inv_mesh : pygimli mesh
        Inversion domain mesh.
    data_list : list of pygimli DataContainerERT
        One container per monitoring epoch, ordered chronologically.
    lambda_s : float, optional
        Spatial regularisation strength.  Default 20.0.
    lambda_t : float, optional
        Temporal coupling base strength.  Default 20.0.
    epsilon : float, optional
        Singularity guard in the weight denominator (log-resistivity units).
        Default 5e-3.
    alpha_max : float, optional
        Hard cap on the per-cell blending coefficient.  Default 0.65.
    max_iter : int, optional
        Maximum number of Gauss-Seidel iterations.  Default 15.
    verbose : bool, optional
        Print per-cell resistivity ranges after each epoch.  Default True.
    """

    def __init__(self, inv_mesh, data_list,
                 lambda_s=20.0, lambda_t=20.0,
                 epsilon=5e-3, alpha_max=0.65,
                 max_iter=15, verbose=True):

        self.mesh      = inv_mesh
        self.data_list = data_list
        self.lambda_s  = lambda_s
        self.lambda_t  = lambda_t
        self.epsilon   = epsilon
        self.alpha_max = alpha_max
        self.max_iter  = max_iter
        self.verbose   = verbose

        self.n_cells = inv_mesh.cellCount()
        self.n_times = len(data_list)

        # Storage updated each iteration
        self.models   = []
        self.weights_t = []
        self.convergence_history = []

        # Spatial graph Laplacian (used only for reference; pyGIMLi handles
        # spatial regularisation internally through ERTManager.invert)
        self.L_s = self._build_laplacian()

        # One ERTManager per epoch, shared across all iterations
        self.managers = []
        for d in data_list:
            mgr = ert.ERTManager()
            mgr.setData(d)
            mgr.setMesh(inv_mesh)
            self.managers.append(mgr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_laplacian(self):
        """Build the sparse graph Laplacian for the inversion mesh."""
        n   = self.n_cells
        nbr = [set() for _ in range(n)]

        for b in self.mesh.boundaries():
            lc = b.leftCell()
            rc = b.rightCell()
            if lc and rc:
                li, ri = lc.id(), rc.id()
                if li < n and ri < n:
                    nbr[li].add(ri)
                    nbr[ri].add(li)

        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i); cols.append(i); vals.append(float(len(nbr[i])))
            for j in nbr[i]:
                rows.append(i); cols.append(j); vals.append(-1.0)

        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    def _compute_weights(self):
        """
        Recompute per-cell temporal weights from the two most recent model
        estimates.  For t < 2 (insufficient history) weights default to 1.
        """
        self.weights_t = []
        for t in range(self.n_times):
            if t < 2:
                self.weights_t.append(np.ones(self.n_cells))
                continue

            diff  = np.abs(self.models[t - 1] - self.models[t - 2])
            w_raw = 1.0 / (self.epsilon + diff)

            med = np.median(w_raw)
            w_norm = w_raw / med if med > 1e-8 else np.ones(self.n_cells)

            p95   = np.percentile(w_norm, 95)
            w_clp = np.clip(w_norm, 0.0, p95)

            self.weights_t.append(w_clp)

    def _invert_step(self, t, current_pass):
        """
        Invert one epoch and blend with the temporal prior.

        Parameters
        ----------
        t : int
            Epoch index.
        current_pass : list of ndarray
            Log-resistivity models produced so far in the current sweep.
            For t == 0 this list is empty; for t > 0 it contains the
            freshly updated estimate of epoch t-1.

        Returns
        -------
        m_blend : ndarray
            Blended log-resistivity vector for epoch t.
        """
        # Warm-start from the previous iteration's estimate when available
        start = None
        if len(self.models) > t:
            start = np.clip(np.exp(self.models[t]), 1.0, 1e4)

        m_sp = self.managers[t].invert(
            self.data_list[t],
            lam=self.lambda_s,
            limits=[1.0, 1e4],
            startModel=start,
            verbose=False,
        )
        m_sp = np.log(np.clip(np.array(m_sp), 1.0, 1e4))

        # No temporal blending for the baseline epoch or when the prior
        # epoch estimate is not yet available in the current pass
        if t > 0 and len(current_pass) >= t:
            lw    = self.lambda_t * self.weights_t[t]
            alpha = np.clip(lw / (self.lambda_s + lw), 0.0, self.alpha_max)
            m_sp  = (1.0 - alpha) * m_sp + alpha * current_pass[t - 1]

        return np.clip(m_sp, 0.0, np.log(1e4))

    def _check_convergence(self, k):
        """
        Apply the windowed plateau convergence criterion.

        Convergence is declared when, over the last CONV_WINDOW iterations:
        (a) the mean update norm has not improved by more than REL_CONV_TOL
            relative to the entry of the window, and
        (b) the coefficient of variation within the window is below CV_TOL.

        Both conditions must hold simultaneously, and at least MIN_ITERS
        iterations must have been completed.

        Returns
        -------
        converged : bool
        diag_str : str  —  diagnostic string (empty if window not yet full)
        """
        hist = self.convergence_history
        if len(hist) < CONV_WINDOW + 1 or k < MIN_ITERS:
            return False, ""

        window    = [h["avg_change"] for h in hist[-CONV_WINDOW:]]
        pre_value = hist[-(CONV_WINDOW + 1)]["avg_change"]
        w_mean    = np.mean(window)
        w_cv      = np.std(window) / w_mean if w_mean > 1e-12 else 0.0
        rel_impr  = (pre_value - w_mean) / pre_value if pre_value > 1e-12 else 0.0

        diag = (
            f"  window_mean={w_mean:.4f}  pre={pre_value:.4f}  "
            f"rel_impr={rel_impr * 100:.2f}%  CV={w_cv * 100:.1f}%"
        )
        converged = rel_impr < REL_CONV_TOL and w_cv < CV_TOL
        return converged, diag

    def _run_one_iteration(self, k):
        """
        Execute one full Gauss-Seidel sweep over all epochs.

        Returns
        -------
        converged : bool
        """
        print(f"\n--- Iteration {k + 1}/{self.max_iter} ---")
        new_models = []

        for t in range(self.n_times):
            print(f"  T={t}...", end="", flush=True)
            m = self._invert_step(t, new_models)
            new_models.append(m)
            rho = np.exp(m)
            if self.verbose:
                print(f" rho=[{rho.min():.1f}, {rho.max():.1f}]")
            else:
                print()

        # Track convergence only when a previous model exists to compare with
        converged = False
        if len(self.models) == len(new_models):
            per_step  = [
                np.linalg.norm(new_models[t] - self.models[t])
                for t in range(self.n_times)
            ]
            # Baseline epoch (t=0) excluded from the averaging metric
            changes   = per_step[1:]
            avg_chg   = float(np.mean(changes))
            self.convergence_history.append({
                "iteration":  k + 1,
                "avg_change": avg_chg,
                "per_step":   per_step,
            })
            print(f"  Avg ||dm|| (T>=1): {avg_chg:.6f}", end="")
            converged, diag = self._check_convergence(k)
            if diag:
                print(f"\n{diag}")
            else:
                print()
            if converged:
                print(f"  Converged at iteration {k + 1} (window={CONV_WINDOW})")

        self.models = new_models
        return converged

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """
        Run HWSI.

        Stage 1 — independent initialisation of all epochs.
        Stage 2 — iterative Gauss-Seidel refinement with adaptive weighting.

        Returns
        -------
        results : list of ndarray
            Inverted resistivity (Ohm.m) at each epoch, in chronological
            order.  Values are bounded to [1, 1e4] Ohm.m.
        """
        print(
            f"\nHWSI  lambda_s={self.lambda_s}  lambda_t={self.lambda_t}  "
            f"eps={self.epsilon}  alpha_max={self.alpha_max}"
        )

        # ---- Stage 1: independent seed ----
        print("\nStage 1: independent initialisation")
        self.models = []
        for t, mgr in enumerate(self.managers):
            m = mgr.invert(
                self.data_list[t],
                lam=self.lambda_s,
                limits=[1.0, 1e4],
                verbose=False,
            )
            self.models.append(np.log(np.clip(np.array(m), 1.0, 1e4)))
            rho = np.exp(self.models[-1])
            print(f"  T={t}: rho=[{rho.min():.1f}, {rho.max():.1f}]")

        self.weights_t = [np.ones(self.n_cells)] * self.n_times

        # ---- Stage 2: adaptive Gauss-Seidel iterations ----
        print(f"\nStage 2: Gauss-Seidel iterations (max {self.max_iter})")
        for k in range(self.max_iter):
            if k > 0:
                self._compute_weights()
            try:
                done = self._run_one_iteration(k)
            except Exception as exc:
                print(f"\nError at iteration {k + 1}: {exc}")
                break
            if done:
                break
        else:
            print("\nReached maximum iterations without declaring convergence.")

        return [np.clip(np.exp(m), 1.0, 1e4) for m in self.models]


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------

def run_independent(inv_mesh, data_list, lam=20.0):
    """
    Independent single-epoch inversion (Daily et al., 1992).

    Each epoch is treated as a standalone problem with no temporal information
    exchanged between timesteps.

    Parameters
    ----------
    inv_mesh  : pygimli mesh
    data_list : list of DataContainerERT
    lam       : float — spatial regularisation strength (default 20.0)

    Returns
    -------
    results : list of ndarray
        Resistivity in Ohm.m at each epoch.
    """
    print(f"\nIndependent inversion  (lam={lam})")
    results = []
    for t, data in enumerate(data_list):
        mgr = ert.ERTManager()
        mgr.setData(data)
        mgr.setMesh(inv_mesh)
        m = mgr.invert(data, lam=lam, limits=[1.0, 1e4], verbose=False)
        m = np.clip(np.array(m), 1.0, 1e4)
        results.append(m)
        print(f"  T={t}: rho=[{m.min():.1f}, {m.max():.1f}]")
    return results


def run_difference(inv_mesh, data_list, lam=20.0):
    """
    Difference inversion (LaBrecque & Yang, 2001).

    Reformulates the problem around change relative to the baseline epoch by
    constructing a transformed data vector  d_diff = (d_t / d_0) * F(m_0),
    where F(m_0) is the forward response of the baseline model.  The
    inversion then recovers resistivity change from baseline.

    Parameters
    ----------
    inv_mesh  : pygimli mesh
    data_list : list of DataContainerERT
    lam       : float — spatial regularisation strength (default 20.0)

    Returns
    -------
    results : list of ndarray
        Resistivity in Ohm.m at each epoch.
    """
    print(f"\nDifference inversion  (lam={lam})")
    results = []

    # Invert baseline epoch independently
    mgr0 = ert.ERTManager()
    mgr0.setData(data_list[0])
    mgr0.setMesh(inv_mesh)
    m0 = np.clip(
        np.array(mgr0.invert(data_list[0], lam=lam, limits=[1.0, 1e4], verbose=False)),
        1.0, 1e4,
    )
    results.append(m0)
    print(f"  T=0 (baseline): rho=[{m0.min():.1f}, {m0.max():.1f}]")

    # Forward response of the baseline model needed for the transformation
    fop0 = ert.ERTModelling()
    fop0.setData(data_list[0])
    fop0.setMesh(inv_mesh, ignoreRegionManager=True)
    f_m0    = np.maximum(np.array(fop0.response(m0)), 0.1)
    rhoa_0  = np.maximum(np.array(data_list[0]["rhoa"]), 0.1)

    for t in range(1, len(data_list)):
        rhoa_t = np.maximum(np.array(data_list[t]["rhoa"]), 0.1)

        # Transformed apparent resistivity data for epoch t
        d_diff = (rhoa_t / rhoa_0) * f_m0

        # Combined error propagated from baseline and current epoch
        err_diff = np.clip(
            np.sqrt(
                np.array(data_list[t]["err"]) ** 2
                + np.array(data_list[0]["err"]) ** 2
            ),
            0.01, 1.0,
        )

        dd = data_list[t].copy()
        dd["rhoa"] = d_diff
        dd.set("err", err_diff)

        mgr = ert.ERTManager()
        mgr.setData(dd)
        mgr.setMesh(inv_mesh)
        mt_ = np.clip(
            np.array(
                mgr.invert(
                    dd, lam=lam, limits=[1.0, 1e4],
                    startModel=m0, isReference=True, verbose=False,
                )
            ),
            1.0, 1e4,
        )
        results.append(mt_)
        print(f"  T={t}: rho=[{mt_.min():.1f}, {mt_.max():.1f}]")

    return results


def run_4d(inv_mesh, data_list, lam=20.0, scalef=1.0):
    """
    4D L2-coupled inversion (Kim et al., 2009) via pyGIMLi's TimelapseERT.

    A spatially uniform temporal smoothness penalty (controlled by scalef)
    is applied identically to every mesh cell and every timestep transition.

    Parameters
    ----------
    inv_mesh  : pygimli mesh
    data_list : list of DataContainerERT
    lam       : float — spatial regularisation strength (default 20.0)
    scalef    : float — temporal coupling strength scale factor (default 1.0)

    Returns
    -------
    results : list of ndarray
        Resistivity in Ohm.m at each epoch.
    """
    print(f"\n4D L2-coupled inversion  (lam={lam}, scalef={scalef})")
    tl = ert.TimelapseERT(DATA=data_list, mesh=inv_mesh)

    # pyGIMLi API changed across minor versions; try the newer signature first
    try:
        tl.fullInversion(scalef=scalef, lam=lam, maxIter=10, verbose=False)
    except TypeError:
        tl.fullInversion(scalef=scalef, lam=lam, verbose=False)

    results = [
        np.clip(np.array(tl.models[t]), 1.0, 1e4)
        for t in range(len(data_list))
    ]
    for t, r in enumerate(results):
        print(f"  T={t}: rho=[{r.min():.1f}, {r.max():.1f}]")
    return results


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def interp_to_grid(mesh, model, gx, gz):
    """
    Interpolate a mesh-based resistivity model onto a regular grid.

    Parameters
    ----------
    mesh  : pygimli mesh
    model : ndarray — per-cell resistivity values
    gx    : ndarray — grid x-coordinates (from np.mgrid)
    gz    : ndarray — grid z-coordinates (from np.mgrid)

    Returns
    -------
    grid_model : ndarray
        Resistivity on the regular grid.  Nodes outside the convex hull of
        the mesh cell centres are set to NaN.
    """
    cc = np.array([[c.center().x(), c.center().y()] for c in mesh.cells()])
    return griddata((cc[:, 0], cc[:, 1]), model, (gx, gz), method="linear")


def build_valid_mask(inv_mesh, gx, gz, grid_extent=(-12, 12)):
    """
    Build a boolean mask that excludes poorly constrained far-corner nodes.

    Nodes are excluded if they lie outside the interpolation support of the
    mesh OR if they are both far from the electrode array (|x| > extent-1.5)
    and deep (z < -4.5 m), where sensitivity is negligible.

    Parameters
    ----------
    inv_mesh    : pygimli mesh
    gx, gz      : ndarray — grid coordinates (from np.mgrid)
    grid_extent : tuple — (xmin, xmax) of the interpolation grid

    Returns
    -------
    mask : ndarray of bool
    """
    dummy = interp_to_grid(inv_mesh, np.ones(inv_mesh.cellCount()), gx, gz)
    far   = (np.abs(gx) > grid_extent[1] - 1.5) & (gz < -4.5)
    return ~np.isnan(dummy) & ~far


def calc_errors(times, true_grids, results_dict, gx, gz, inv_mesh, mask=None):
    """
    Compute RMSE, normalised RMSE and MAE for a set of inversion results.

    Parameters
    ----------
    times        : array-like — epoch times (hours)
    true_grids   : list of ndarray — true resistivity on the regular grid
    results_dict : dict {method_name: list_of_per_cell_resistivity_arrays}
    gx, gz       : ndarray — grid coordinates
    inv_mesh     : pygimli mesh
    mask         : ndarray of bool, optional — valid node mask

    Returns
    -------
    errors : dict
        {method_name: {"rmse": [...], "nrmse": [...], "mae": [...]}}
    """
    errors = {m: {"rmse": [], "nrmse": [], "mae": []} for m in results_dict}

    for ti in range(len(times)):
        tg  = true_grids[ti]
        rng = np.max(tg) - np.min(tg)
        # Use mean as normaliser when the dynamic range is negligible (e.g. t=0)
        nom = np.mean(tg) if rng < 1.0 else rng

        for name, res in results_dict.items():
            ig    = interp_to_grid(inv_mesh, res[ti], gx, gz)
            valid = (~np.isnan(ig)) if mask is None else (mask & ~np.isnan(ig))

            if valid.any():
                d = tg[valid] - ig[valid]
                errors[name]["rmse"].append(float(np.sqrt(np.mean(d ** 2))))
                errors[name]["nrmse"].append(
                    float(np.sqrt(np.mean(d ** 2)) / nom * 100)
                    if nom > 1e-8 else float("nan")
                )
                errors[name]["mae"].append(float(np.mean(np.abs(d))))
            else:
                for k in errors[name]:
                    errors[name][k].append(float("nan"))

    return errors
