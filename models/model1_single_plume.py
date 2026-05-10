"""
model1_single_plume.py
----------------------
Spatiotemporal ERT Inversion — Model 1: Single Centred Conductive Plume

Reproduces all figures and the nRMSE comparison table for Model 1 from
the HWSI paper.  Run this script directly:

    python models/model1_single_plume.py

Outputs (saved to the working directory)
-----------------------------------------
model1_hwsi_true_models.tiff
model1_hwsi_inversion_models.tiff
model1_hwsi_error_analysis.tiff
model1_hwsi_true_temporal_changes.tiff
model1_hwsi_inversion_temporal_changes.tiff
model1_hwsi_convergence.tiff

The nRMSE table is also printed to the terminal.
"""

import math
import sys
import os

# Allow running from the repository root or from the models/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

from hwsi import (
    HWSI_Inversion,
    run_independent,
    run_difference,
    run_4d,
    interp_to_grid,
    build_valid_mask,
    calc_errors,
)


# ---------------------------------------------------------------------------
# Global figure style
# ---------------------------------------------------------------------------

FONT_BOLD = True
FONT_SIZE  = 25

_fw  = "bold" if FONT_BOLD else "normal"
_fss = max(FONT_SIZE - 5, 6)   # tick labels, legend
_fsn = FONT_SIZE                # axis labels
_fst = FONT_SIZE + 2            # subplot titles

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          _fss,
    "font.weight":        _fw,
    "axes.titlesize":     _fsn,
    "axes.titleweight":   _fw,
    "axes.labelsize":     _fsn,
    "axes.labelweight":   _fw,
    "xtick.labelsize":    _fss,
    "ytick.labelsize":    _fss,
    "legend.fontsize":    _fss,
    "figure.titlesize":   _fsn,
    "figure.titleweight": _fw,
    "mathtext.default":   "regular",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})


# ---------------------------------------------------------------------------
# Spatial / grid constants
# ---------------------------------------------------------------------------

COMPARISON_CROP_PCT = 100
COMPARISON_X_RANGE  = (-9, 9)
TEMPORAL_CROP_PCT   = 0
TEMPORAL_X_RANGE    = (-12, 12)
GRID_EXTENT         = (-12, 12)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
# Two meshes with different topology and extent to avoid the inverse crime.
# Forward mesh: large domain, coarser resolution.
# Inversion mesh: restricted domain, finer resolution.

scheme = ert.createData(elecs=np.linspace(-12, 12, 50), schemeName="dd")

world = mt.createWorld(start=[-25, -15], end=[25, 0], worldMarker=True)
for p in scheme.sensorPositions():
    world.createNode(p)
fwd_mesh = mt.createMesh(world, quality=34.5, area=0.2)

inv_domain = mt.createRectangle(start=[-12, -6], end=[12, 0])
inv_mesh   = mt.createMesh(inv_domain, quality=34.5, area=0.1)

# Regular interpolation grid for visualisation and error evaluation
grid_x, grid_z = np.mgrid[GRID_EXTENT[0]:GRID_EXTENT[1]:300j, -6:0:150j]

print(f"Forward mesh : {fwd_mesh.cellCount()} cells")
print(f"Inversion mesh: {inv_mesh.cellCount()} cells")


# ---------------------------------------------------------------------------
# True resistivity model — single expanding conductive plume
# ---------------------------------------------------------------------------

NOISE_LEVEL = 0.03    # fractional Gaussian noise added to synthetic data
DATA_ERROR  = 0.05    # conservative uniform data error assumed in inversion
BG_RHO      = 600.0   # background resistivity (Ohm.m)

R_MIN     = 0.5       # plume radius at t=0 h (m)
R_MAX     = 4.0       # plume radius at t=24 h (m)
PLUME_RHO = 10.0      # plume resistivity (Ohm.m)
PLUME_CX  = 0.0       # plume centre x (m)
PLUME_CZ  = 0.0       # plume centre z (m, 0 = surface)


def _plume_radius(t_hour):
    """Linear growth of the plume radius with time."""
    return R_MIN + (R_MAX - R_MIN) * (t_hour / 24.0)


def get_true_resistivity_grid(gx, gz, t_hour):
    """
    Return the true resistivity on the regular grid at time t_hour.

    The plume is a semicircle centred at the surface (z=0) that grows
    linearly in radius.  Only the subsurface half (z <= 0) is filled.
    """
    rho = np.full(gx.shape, BG_RHO)
    if t_hour <= 0:
        return rho
    rad  = _plume_radius(t_hour)
    dist = np.sqrt((gx - PLUME_CX) ** 2 + (gz - PLUME_CZ) ** 2)
    rho[(dist <= rad) & (gz <= 0)] = PLUME_RHO
    return rho


def get_true_resistivity(x, z, t_hour):
    """Return the true resistivity at a single point and time."""
    if z > 0 or t_hour <= 0:
        return BG_RHO
    rad  = _plume_radius(t_hour)
    dist = np.sqrt((x - PLUME_CX) ** 2 + (z - PLUME_CZ) ** 2)
    return PLUME_RHO if dist <= rad else BG_RHO


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

times = np.arange(0, 25, 4)   # 0, 4, 8, 12, 16, 20, 24 h
data_list  = []
true_grids = []

print("\nGenerating synthetic data...")
for t in times:
    print(f"  T={t:2d}h", end="", flush=True)

    # Assign true resistivity to every forward-mesh cell
    rho_vec = np.array([
        get_true_resistivity(c.center().x(), c.center().y(), t)
        for c in fwd_mesh.cells()
    ])

    # Forward simulation with noise
    data = ert.simulate(
        mesh=fwd_mesh,
        scheme=scheme,
        res=rho_vec,
        noiseLevel=NOISE_LEVEL,
        seed=123,
        verbose=False,
    )
    data["rhoa"] = np.maximum(np.abs(data["rhoa"]), 0.1)
    data.set("err", np.full(data.size(), DATA_ERROR))

    data_list.append(data)
    tg = get_true_resistivity_grid(grid_x, grid_z, t)
    true_grids.append(tg)

    plume_count = int(np.sum(tg < 50))
    r_now       = _plume_radius(t) if t > 0 else 0.0
    print(f" | plume cells: {plume_count:5d} | r={r_now:.2f} m")


# ---------------------------------------------------------------------------
# Inversions
# ---------------------------------------------------------------------------

LAMBDA_S = 20.0
LAMBDA_T = 20.0
SCALEF   = 1.0

results_independent = run_independent(inv_mesh, data_list, lam=LAMBDA_S)
results_difference  = run_difference(inv_mesh, data_list, lam=LAMBDA_S)
results_4d          = run_4d(inv_mesh, data_list, lam=LAMBDA_S, scalef=SCALEF)

hwsi = HWSI_Inversion(
    inv_mesh, data_list,
    lambda_s=LAMBDA_S, lambda_t=LAMBDA_T,
    epsilon=5e-3, alpha_max=0.65,
    max_iter=15, verbose=True,
)
results_hwsi = hwsi.run()

results_dict = {
    "Independent":   results_independent,
    "Difference":    results_difference,
    "4D L2-Coupled": results_4d,
    "HWSI":          results_hwsi,
}


# ---------------------------------------------------------------------------
# Error evaluation
# ---------------------------------------------------------------------------

valid_mask = build_valid_mask(inv_mesh, grid_x, grid_z, grid_extent=GRID_EXTENT)
errors     = calc_errors(
    times, true_grids, results_dict,
    grid_x, grid_z, inv_mesh, mask=valid_mask,
)

baseline = np.nanmean(errors["Independent"]["nrmse"])

print(f"\n{'Method':<20} {'Mean nRMSE':>11} {'Std':>7} {'vs Indep':>10}")
print("-" * 52)
for mn in results_dict:
    avg = np.nanmean(errors[mn]["nrmse"][1:])
    std = np.nanstd(errors[mn]["nrmse"][1:])
    print(f"{mn:<20} {avg:>10.2f}% {std:>6.2f}%  {baseline - avg:>+8.2f} pp")

print(f"\n{'Time':<8}", end="")
for mn in results_dict:
    print(f"{mn:<18}", end="")
print()
for ti, t in enumerate(times):
    if t == 0:
        continue
    print(f"T={t:2d}h   ", end="")
    for mn in results_dict:
        print(f"{errors[mn]['nrmse'][ti]:>7.2f}%       ", end="")
    print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Independent":   "#1f77b4",
    "Difference":    "#ff7f0e",
    "4D L2-Coupled": "#d62728",
    "HWSI":          "#2ca02c",
}


def _roman(n):
    """Convert integer to lowercase Roman numeral string."""
    _vals = [
        (1000,"m"),(900,"cm"),(500,"d"),(400,"cd"),
        (100,"c"),(90,"xc"),(50,"l"),(40,"xl"),
        (10,"x"),(9,"ix"),(5,"v"),(4,"iv"),(1,"i"),
    ]
    r = ""
    for v, s in _vals:
        while n >= v:
            r += s; n -= v
    return r


def _crop(data, pct=30, xr=(-7, 7), ge=(-12, 12)):
    """
    Apply a tapered lateral crop to a transposed image array.

    Rows near the top (shallow depth) are not cropped; rows near the bottom
    (deeper) are progressively masked to remove poorly constrained far-corner
    artefacts.
    """
    d     = data.copy()
    h, w  = d.shape
    gx0, gx1 = ge
    c0 = int((xr[0] - gx0) / (gx1 - gx0) * w)
    c1 = int((xr[1] - gx0) / (gx1 - gx0) * w)
    f  = pct / 100.0

    for i in range(h):
        e   = ((h - 1 - i) / (h - 1) if h > 1 else 0) * f
        lo  = int((1 - e) * 0 + e * c0)
        hi  = int((1 - e) * w + e * c1)
        if lo > 0:
            d[i, :lo] = np.nan
        if hi < w:
            d[i, hi:] = np.nan
    return d


def _jet_white():
    """
    Jet colormap with a white band centred at zero for temporal change maps.
    """
    jet = plt.cm.jet
    anchors = [
        (0.00, jet(0.00)), (0.39, jet(0.39)),
        (0.47, (1, 1, 1, 1)), (0.55, (1, 1, 1, 1)),
        (0.61, jet(0.61)), (1.00, jet(1.00)),
    ]
    colors = []
    for p in np.linspace(0, 1, 256):
        for k in range(len(anchors) - 1):
            p0, c0 = anchors[k]
            p1, c1 = anchors[k + 1]
            if p0 <= p <= p1:
                t_frac = (p - p0) / (p1 - p0) if p1 > p0 else 0
                colors.append(
                    tuple((1 - t_frac) * c0[i] + t_frac * c1[i] for i in range(4))
                )
                break
    return LinearSegmentedColormap.from_list("jet_white", colors, N=256)


def _style_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight(_fw)
        lbl.set_fontsize(_fss)


def _bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


# ---------------------------------------------------------------------------
# Figure 1 — True models
# ---------------------------------------------------------------------------

def plot_true_models(times, true_grids):
    norm = Normalize(vmin=0, vmax=4)
    fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 4.5))

    for ti, t in enumerate(times):
        ax  = axes[ti]
        img = _crop(
            np.log10(np.clip(true_grids[ti], 1, 1e4)).T,
            COMPARISON_CROP_PCT, COMPARISON_X_RANGE, GRID_EXTENT,
        )
        ax.imshow(
            img, cmap="jet", norm=norm, origin="lower",
            extent=[GRID_EXTENT[0], GRID_EXTENT[1], -6, 0],
            aspect="equal", interpolation="bilinear",
        )
        ax.set_title(f"True Model\nt={t}h", fontweight=_fw, fontsize=_fst)
        if ti == 0:
            ax.set_ylabel("Depth (m)", fontweight=_fw, fontsize=_fsn)
        ax.set_xlabel("Distance (m)", fontweight=_fw, fontsize=_fsn)
        _style_ticks(ax)

    bg_col    = plt.cm.jet(norm(np.log10(BG_RHO)))
    plume_col = plt.cm.jet(norm(np.log10(PLUME_RHO)))
    patches   = [
        plt.Rectangle((0, 0), 1, 1, color=bg_col),
        plt.Rectangle((0, 0), 1, 1, color=plume_col),
    ]
    labels = [
        f"Background: {int(BG_RHO)} Ohm.m",
        f"Plume: {int(PLUME_RHO)} Ohm.m (conductive)",
    ]
    fig.subplots_adjust(bottom=0.18, wspace=0.12, left=0.05, right=0.98, top=0.93)
    fig.legend(
        patches, labels, loc="lower center", ncol=2, frameon=True,
        handlelength=2.5, handleheight=1.5,
        prop={"weight": _fw, "size": _fst},
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Inversion model panels
# ---------------------------------------------------------------------------

def plot_inversion_models(times, results_dict, gx, gz, mesh):
    methods = list(results_dict.keys())
    norm    = Normalize(vmin=0, vmax=4)
    fig, axes = plt.subplots(
        len(methods), len(times),
        figsize=(5 * len(times), 4.5 * len(methods)),
    )

    for ti, t in enumerate(times):
        for mi, mn in enumerate(methods):
            ax  = axes[mi, ti]
            gd  = interp_to_grid(mesh, results_dict[mn][ti], gx, gz)
            img = _crop(
                np.log10(np.clip(gd, 1, 1e4)).T,
                COMPARISON_CROP_PCT, COMPARISON_X_RANGE, GRID_EXTENT,
            )
            ax.imshow(
                img, cmap="jet", norm=norm, origin="lower",
                extent=[GRID_EXTENT[0], GRID_EXTENT[1], -6, 0],
                aspect="equal", interpolation="bilinear",
            )
            ax.set_title(
                f"{_roman(mi + 1)})  {mn}\nt={t}h",
                fontweight=_fw, fontsize=_fst,
            )
            if ti == 0:
                ax.set_ylabel("Depth (m)", fontweight=_fw, fontsize=_fsn)
            ax.set_xlabel("Distance (m)", fontweight=_fw, fontsize=_fsn)
            _style_ticks(ax)

    fig.subplots_adjust(right=0.91, hspace=0.0, wspace=0.12,
                        left=0.05, top=0.93, bottom=0.05)
    cb = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    c  = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), cax=cb)
    c.set_ticks([0, 1, 2, 3, 4])
    c.set_ticklabels(["0", "1", "2", "3", "4"])
    c.set_label("log10(rho) [Ohm.m]", fontweight=_fw, fontsize=_fsn + 5)
    for lbl in c.ax.get_yticklabels():
        lbl.set_fontweight(_fw); lbl.set_fontsize(_fss + 5)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — nRMSE error trajectories
# ---------------------------------------------------------------------------

def plot_errors(times, errors):
    fig, ax = plt.subplots(figsize=(11, 6))
    xt = range(1, len(times))

    for mn in errors:
        ax.plot(
            xt, errors[mn]["nrmse"][1:],
            linewidth=2.5, label=mn,
            color=METHOD_COLORS.get(mn, "gray"),
        )

    ax.set_xlabel("Time step", fontweight="bold", fontsize=_fsn - 5)
    ax.set_ylabel("nRMSE (%)", fontweight="bold", fontsize=_fsn - 5)
    ax.set_xticks(list(xt))
    ax.set_xticklabels(
        [f"T={t}h" for t in times[1:]],
        fontweight="bold", fontsize=_fss - 5,
    )
    ax.set_ylim(20, 100)
    ax.set_yticks(range(20, 101, 10))
    ax.set_yticklabels(
        [str(y) for y in range(20, 101, 10)],
        fontweight="bold", fontsize=_fss - 5,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(prop={"weight": "bold", "size": _fss - 6})
    _bold_ticks(ax)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — True temporal change maps
# ---------------------------------------------------------------------------

def plot_true_temporal_change(times, true_grids, threshold=0.12):
    """
    Log-resistivity change between consecutive true-model epochs.
    TwoSlopeNorm anchors white at zero; the extended negative axis resolves
    the strongly conductive plume signal (log10(10/600) ≈ -1.78).
    """
    ntrans  = len(times) - 1
    cmap    = _jet_white()
    norm_tc = TwoSlopeNorm(vcenter=0.0, vmin=-1.8, vmax=0.6)

    fig = plt.figure(figsize=(6.5 * ntrans, 5.0))
    gs  = GridSpec(
        1, ntrans, figure=fig,
        hspace=0.0, wspace=0.12,
        left=0.05, right=0.98, top=0.88, bottom=0.22,
    )

    for ti in range(ntrans):
        tp, tc = times[ti], times[ti + 1]
        ax     = fig.add_subplot(gs[0, ti])
        chg    = np.log10(
            np.clip(true_grids[ti + 1], 1, 1e4) /
            np.clip(true_grids[ti],     1, 1e4)
        )
        img = _crop(
            np.where(np.abs(chg) < threshold, np.nan, chg).T,
            TEMPORAL_CROP_PCT, TEMPORAL_X_RANGE, GRID_EXTENT,
        )
        ax.imshow(
            img, cmap=cmap, norm=norm_tc, origin="lower",
            extent=[GRID_EXTENT[0], GRID_EXTENT[1], -6, 0],
            aspect="equal", interpolation="bilinear", rasterized=True,
        )
        ax.set_facecolor("white")
        ax.set_title(f"True\n{tp}h->{tc}h", fontweight=_fw, fontsize=_fst, pad=10)
        if ti == 0:
            ax.set_ylabel("Depth (m)", fontweight=_fw, fontsize=_fsn)
        ax.set_xlabel("Distance (m)", fontweight=_fw, fontsize=_fsn)
        _style_ticks(ax)

    col_plume = cmap(norm_tc(-1.78))
    patches   = [
        plt.Rectangle((0, 0), 1, 1, color="white", ec="gray", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, color=col_plume),
    ]
    labels = [
        "No change (stable)",
        f"Plume — strong decrease (~{int(PLUME_RHO)} Ohm.m)",
    ]
    fig.legend(
        patches, labels, loc="lower center", ncol=2, frameon=True,
        handlelength=2.5, handleheight=1.5,
        prop={"weight": _fw, "size": _fst},
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Inverted temporal change maps
# ---------------------------------------------------------------------------

def plot_inversion_temporal_change(times, results_dict, gx, gz, mesh,
                                   threshold=0.12):
    methods = list(results_dict.keys())
    ntrans  = len(times) - 1
    cmap    = _jet_white()

    fig = plt.figure(figsize=(6.5 * ntrans, 4.5 * len(methods)))
    gs  = GridSpec(
        len(methods), ntrans, figure=fig,
        hspace=0.18, wspace=0.12,
        left=0.05, right=0.87, top=0.93, bottom=0.08,
    )

    for ti in range(ntrans):
        tp, tc = times[ti], times[ti + 1]
        for mi, mn in enumerate(methods):
            ax  = fig.add_subplot(gs[mi, ti])
            rp  = np.clip(interp_to_grid(mesh, results_dict[mn][ti],     gx, gz), 1, 1e4)
            rc  = np.clip(interp_to_grid(mesh, results_dict[mn][ti + 1], gx, gz), 1, 1e4)
            chg = np.log10(rc / rp)
            img = _crop(
                np.where(np.abs(chg) < threshold, np.nan, chg).T,
                TEMPORAL_CROP_PCT, TEMPORAL_X_RANGE, GRID_EXTENT,
            )
            ax.imshow(
                img, cmap=cmap, vmin=-0.6, vmax=0.6, origin="lower",
                extent=[GRID_EXTENT[0], GRID_EXTENT[1], -6, 0],
                aspect="equal", interpolation="bilinear", rasterized=True,
            )
            ax.set_facecolor("white")
            ax.set_title(
                f"{_roman(mi + 1)})  {mn}\n{tp}h->{tc}h",
                fontweight=_fw, fontsize=_fst,
            )
            if ti == 0:
                ax.set_ylabel("Depth (m)", fontweight=_fw, fontsize=_fsn)
            ax.set_xlabel("Distance (m)", fontweight=_fw, fontsize=_fsn)
            _style_ticks(ax)

    cb = fig.add_axes([0.88, 0.25, 0.015, 0.50])
    c  = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-0.6, vmax=0.6), cmap=cmap),
        cax=cb,
    )
    c.set_label(
        r"$\log_{10}(\rho_t\,/\,\rho_{t-1})$",
        labelpad=15, fontweight=_fw, fontsize=_fsn + 5,
    )
    for lbl in c.ax.get_yticklabels():
        lbl.set_fontweight(_fw); lbl.set_fontsize(_fss + 5)
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — HWSI convergence curve
# ---------------------------------------------------------------------------

def plot_convergence(hwsi_inst):
    hist = hwsi_inst.convergence_history
    if not hist:
        print("No convergence history available — skipping convergence plot.")
        return None

    iters = [h["iteration"]  for h in hist]
    avgs  = [h["avg_change"] for h in hist]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.91)

    ax.semilogy(iters, avgs, color="#2ca02c", linewidth=2.6,
                solid_capstyle="round", zorder=3)

    exp_lo = math.floor(math.log10(min(avgs)))
    exp_hi = math.floor(math.log10(max(avgs)))
    ax.set_yticks([10 ** e for e in range(exp_lo, exp_hi + 1)])
    ax.set_yticklabels(
        [f"$10^{{{e}}}$" for e in range(exp_lo, exp_hi + 1)],
        fontweight=_fw, fontsize=_fss - 5,
    )
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(10 ** (exp_lo - 0.15), 10 ** (exp_hi + 0.25))

    ax.set_xticks(iters)
    ax.set_xticklabels([str(i) for i in iters], fontweight=_fw, fontsize=_fss - 5)
    ax.set_xlim(iters[0] - 0.5, iters[-1] + 0.5)

    ax.set_xlabel("Iteration",            fontweight=_fw, fontsize=_fsn - 6)
    ax.set_ylabel(r"Avg $||\Delta m||$",  fontweight=_fw, fontsize=_fsn - 6)

    ax.grid(True, which="major", color="#cccccc", linestyle="-",
            linewidth=0.5, zorder=0)
    ax.grid(False, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(which="major", direction="in", length=4, width=0.8,
                   top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=2, width=0.6,
                   top=True, right=True)
    return fig


# ---------------------------------------------------------------------------
# Save all figures
# ---------------------------------------------------------------------------

print("\nSaving figures...")
output_dir = os.path.dirname(os.path.abspath(__file__))

figure_jobs = [
    (
        plot_true_models(times, true_grids),
        "model1_hwsi_true_models.tiff",
    ),
    (
        plot_inversion_models(times, results_dict, grid_x, grid_z, inv_mesh),
        "model1_hwsi_inversion_models.tiff",
    ),
    (
        plot_errors(times, errors),
        "model1_hwsi_error_analysis.tiff",
    ),
    (
        plot_true_temporal_change(times, true_grids),
        "model1_hwsi_true_temporal_changes.tiff",
    ),
    (
        plot_inversion_temporal_change(
            times, results_dict, grid_x, grid_z, inv_mesh
        ),
        "model1_hwsi_inversion_temporal_changes.tiff",
    ),
    (
        plot_convergence(hwsi),
        "model1_hwsi_convergence.tiff",
    ),
]

for fig, fname in figure_jobs:
    if fig is not None:
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="tiff")
        print(f"  Saved: {fname}")

plt.show()
