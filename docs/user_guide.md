# HWSI User Guide



## Contents

1. [Overview](#1-overview)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [HWSI_Inversion — full reference](#5-hwsi_inversion--full-reference)
6. [Baseline methods](#6-baseline-methods)
7. [Utility functions](#7-utility-functions)
8. [Plotting and figures](#8-plotting-and-figures)
9. [Key parameters — sensitivity guide](#9-key-parameters--sensitivity-guide)
10. [Reproducing the paper tables](#10-reproducing-the-paper-tables)
11. [Adapting HWSI to new datasets](#11-adapting-hwsi-to-new-datasets)
12. [Common issues and fixes](#12-common-issues-and-fixes)

---

## 1. Overview

HWSI (History-Weighted Spatiotemporal Inversion) is a time-lapse ERT inversion framework that addresses the following problem: any spatially uniform temporal coupling strong enough to suppress noise in stable background cells is also strong enough to damp the response of cells that are genuinely changing. HWSI resolves this by assigning each mesh cell its own temporal coupling coefficient, derived from the two most recently accepted model estimates. Stable cells receive large coefficients and are strongly anchored to their temporal prior; changing cells receive small coefficients and are left free to follow the current data.

The weight field is recomputed at every Gauss-Seidel iteration, so it co-evolves with the inversion solution. No preliminary estimation step is required and only two model epochs need to be held in memory at any time, regardless of how long the monitoring sequence is.

---

## 2. Repository Layout

```
hwsi_repo/
│
├── LICENSE                      MIT licence
├── README.md                    Installation, quick start, expected outputs
├── requirements.txt             Python package versions
│
├── hwsi/
│   ├── __init__.py              Public API exports
│   └── hwsi.py                  HWSI class + all baseline methods + utilities
│
├── models/
│   ├── model1_single_plume.py   End-to-end script for the single-plume scenario
│   └── model2_two_plume.py      End-to-end script for the dual-plume scenario
│
├── notebooks/
│   ├── 01_model1_tutorial.ipynb Step-by-step notebook for Model 1
│   └── 02_model2_tutorial.ipynb Step-by-step notebook for Model 2
│
└── docs/
    └── user_guide.md            This document
```

---

## 3. Installation

### Requirements

| Software | Version |
|---|---|
| Python | 3.9 or later |
| pyGIMLi | 1.5.5 |
| NumPy | 1.23 or later |
| SciPy | 1.9 or later |
| Matplotlib | 3.6 or later |

### Step-by-step

**Step 1 — clone the repository**

```bash
git clone https://github.com/<your-username>/hwsi.git
cd hwsi
```

**Step 2 — create a virtual environment**

Using conda (recommended because pyGIMLi links against compiled C++ libraries):

```bash
conda create -n hwsi_env -c gimli -c conda-forge pygimli=1.5.5
conda activate hwsi_env
```

Or using pip with a virtual environment:

```bash
python -m venv hwsi_env
source hwsi_env/bin/activate    # Linux / macOS
hwsi_env\Scripts\activate       # Windows
```

**Step 3 — install remaining dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — verify the installation**

```python
import pygimli
import numpy, scipy, matplotlib
print(pygimli.__version__)   # should print 1.5.5
```

---

## 4. Quick Start

### Running a model script

```bash
python models/model1_single_plume.py
python models/model2_two_plume.py
```

Each script runs the complete workflow: mesh setup → data generation → four inversions → error tables → figure export.  Runtime is dominated by the inversion step; on a modern workstation expect 30–90 minutes per script depending on CPU speed.

### Using the HWSI class directly

```python
import numpy as np
from pygimli.physics import ert
from hwsi import HWSI_Inversion

# --- Build mesh and data_list from your own survey ---
# inv_mesh  : pyGIMLi mesh covering the inversion domain
# data_list : list of pyGIMLi DataContainerERT, one per epoch

hwsi = HWSI_Inversion(
    inv_mesh, data_list,
    lambda_s=20.0,
    lambda_t=20.0,
    epsilon=5e-3,
    alpha_max=0.65,
    max_iter=15,
    verbose=True,
)
results = hwsi.run()   # list of resistivity arrays, one per epoch
```

### Running a notebook

```bash
jupyter notebook notebooks/01_model1_tutorial.ipynb
```

---

## 5. HWSI_Inversion — Full Reference

### Class signature

```python
HWSI_Inversion(
    inv_mesh,
    data_list,
    lambda_s = 20.0,
    lambda_t = 20.0,
    epsilon  = 5e-3,
    alpha_max = 0.65,
    max_iter  = 15,
    verbose   = True,
)
```

### Parameters

**inv_mesh** *(pyGIMLi mesh)*  
The inversion domain mesh.  Must be a pyGIMLi `Mesh` object.  A finer mesh improves spatial resolution but increases computational cost linearly.

**data_list** *(list of DataContainerERT)*  
One container per monitoring epoch, in chronological order.  Each container must have `rhoa` (apparent resistivity) and `err` (fractional data error) arrays set.

**lambda_s** *(float, default 20.0)*  
Spatial regularisation strength passed to pyGIMLi's `ERTManager.invert`.  Controls how strongly the spatial Laplacian smoothing penalty is applied.  Higher values produce smoother models but reduce sensitivity to small anomalies.  See Section 9 for guidance on choosing this value.

**lambda_t** *(float, default 20.0)*  
Temporal coupling base strength.  Together with the per-cell weight w, this controls the blending coefficient alpha = lambda_t * w / (lambda_s + lambda_t * w).  When lambda_t equals lambda_s, a cell with weight w = 1 (the median weight) receives alpha = 0.5.

**epsilon** *(float, default 5e-3)*  
Singularity guard in the weight denominator.  Prevents division by zero when two consecutive model estimates are identical.  Units are log-resistivity.  The method is broadly insensitive to this parameter across the range 1e-4 to 1e-2 (see paper Table 8).  No tuning is required in practice.

**alpha_max** *(float, default 0.65)*  
Hard upper cap on the per-cell blending coefficient.  Ensures that even the most stable cells remain responsive to the current dataset and cannot be locked entirely to their temporal prior.  Values below approximately 0.55 degrade performance because stable background cells are not anchored firmly enough to resist noise.  Values above 0.65 generally improve accuracy further in synthetic tests but should be treated with caution in field applications where the subsurface may evolve less predictably.

**max_iter** *(int, default 15)*  
Maximum number of Gauss-Seidel iterations.  The windowed plateau convergence criterion usually triggers well before this limit.  Increase if convergence is not declared within 15 iterations (rare in practice).

**verbose** *(bool, default True)*  
Print per-epoch resistivity ranges after each epoch during the iterative phase.

### Attributes (after calling run())

**models** *(list of ndarray)*  
Log-resistivity vectors at each epoch from the final iteration.

**weights_t** *(list of ndarray)*  
Per-cell temporal weight vectors from the final iteration.

**convergence_history** *(list of dict)*  
One entry per completed iteration.  Each entry contains:
- `'iteration'` : iteration number (1-indexed)
- `'avg_change'` : mean update norm across all epochs except t=0
- `'per_step'` : list of per-epoch update norms

### Method: run()

Executes the two-stage HWSI algorithm and returns the final resistivity models.

**Returns**  
`results` : list of ndarray, length equal to the number of epochs.  Each array contains per-cell resistivity in Ωm, bounded to [1, 1e4].

### Convergence criterion

Convergence is declared when both of the following hold over a sliding window of three consecutive iterations, no earlier than iteration 4:

- The relative improvement in the mean update norm is below 3%.  A slight rise in the norm (negative improvement) also satisfies this condition, correctly identifying a plateau.
- The coefficient of variation within the window is below 0.15, guarding against premature termination at transient dips.

---

## 6. Baseline Methods

All three baseline functions share the same interface pattern.

### run_independent

```python
run_independent(inv_mesh, data_list, lam=20.0)
```

Inverts each epoch independently using pyGIMLi's `ERTManager`.  No information passes between epochs.  Serves as the no-coupling baseline.

**Returns** list of per-cell resistivity arrays (Ωm).

### run_difference

```python
run_difference(inv_mesh, data_list, lam=20.0)
```

Implements difference inversion following LaBrecque & Yang (2001).  The baseline epoch is inverted independently; subsequent epochs are inverted against a transformed data vector that encodes resistivity change from baseline.  Works well for small perturbations but degrades as cumulative change grows large.

**Returns** list of per-cell resistivity arrays (Ωm).

### run_4d

```python
run_4d(inv_mesh, data_list, lam=20.0, scalef=1.0)
```

Wraps pyGIMLi's `TimelapseERT.fullInversion` for 4D L2-coupled inversion (Kim et al., 2009).  A spatially uniform temporal penalty (scaled by `scalef`) is applied to all cells equally.

**Parameter `scalef`**: temporal coupling scale factor.  The paper uses `scalef=1.0`.

**Returns** list of per-cell resistivity arrays (Ωm).

---

## 7. Utility Functions

### interp_to_grid

```python
interp_to_grid(mesh, model, gx, gz)
```

Interpolates a per-cell mesh model onto a regular grid using `scipy.interpolate.griddata` with linear interpolation.  Nodes outside the convex hull of the mesh cell centres are set to NaN.

**Parameters**
- `mesh` : pyGIMLi mesh
- `model` : ndarray of per-cell values
- `gx`, `gz` : 2-D coordinate arrays from `np.mgrid`

**Returns** 2-D ndarray on the regular grid.

### build_valid_mask

```python
build_valid_mask(inv_mesh, gx, gz, grid_extent=(-12, 12))
```

Constructs a boolean mask that excludes nodes outside the mesh interpolation support and nodes in the far corners where ERT sensitivity is negligible (|x| > extent−1.5 m and z < −4.5 m).

**Returns** 2-D boolean ndarray.

### calc_errors

```python
calc_errors(times, true_grids, results_dict, gx, gz, inv_mesh, mask=None)
```

Computes RMSE, normalised RMSE and MAE for each method at each epoch.

**Parameters**
- `times` : array of epoch times
- `true_grids` : list of true-resistivity 2-D arrays on the regular grid
- `results_dict` : `{method_name: [per-cell arrays]}`
- `mask` : optional boolean mask from `build_valid_mask`

**Returns**  
Nested dict `{method_name: {'rmse': [...], 'nrmse': [...], 'mae': [...]}}` where each inner list has one value per epoch.

The normalised RMSE is defined as  
nRMSE(t) = RMSE(t) / R(t) × 100%  
where R(t) = max(ρ_true) − min(ρ_true) is the dynamic range of the true model at epoch t.

---

## 8. Plotting and Figures

Both model scripts contain six plotting functions that produce publication-quality TIFF figures at 300 dpi.

| Function | Output figure |
|---|---|
| `plot_true_models` | True resistivity panels across all epochs |
| `plot_inversion_models` | Inverted resistivity — all methods × all epochs |
| `plot_errors` | nRMSE trajectories |
| `plot_true_temporal_change` | True log-ratio change maps |
| `plot_inversion_temporal_change` | Inverted change maps — all methods |
| `plot_convergence` | HWSI mean update norm vs. iteration |

Global font settings (`FONT_BOLD`, `FONT_SIZE`) near the top of each model script control the appearance of all figures uniformly.

---

## 9. Key Parameters — Sensitivity Guide

This section summarises findings from the sensitivity experiments in the paper (all using Model 1).

### lambda_s and lambda_t

HWSI leads at 6 of 9 combinations in the 3 × 3 grid {10, 20, 40} × {10, 20, 40}.  The advantage is most consistent at moderate values (lambda_s = 10–20).  At lambda_s = 40, aggressive spatial smoothing dominates before temporal coupling can engage and the margin over 4D L2-coupled shrinks.

**Recommendation**: start with lambda_s = lambda_t = 20 and adjust together.

### alpha_max

Performance degrades below alpha_max ≈ 0.55 because stable background cells cannot be anchored firmly enough.  Above 0.55 the method leads consistently; above 0.65 accuracy continues to improve in the synthetic tests but the benefit is small.

**Recommendation**: use alpha_max = 0.65 as the default.  In field settings with uncertain background behaviour, consider values in the range 0.55–0.65.

### epsilon

The method is broadly insensitive to epsilon across 1e-4 to 1e-2.  The 95th-percentile clipping largely prevents outlier instability at very small values.

**Recommendation**: leave at the default 5e-3.  No tuning is needed.

### Noise and data error

At matched noise/error levels of 3%/5% (the paper default), HWSI leads by approximately 10 percentage points over Independent inversion (Model 1).  The margin narrows slightly at 7%/7% as aggressive data distrust causes spatial smoothing to dominate.

**Recommendation**: use a data error floor modestly above the actual noise level (e.g., noise = 3%, error = 5%), following Lesparre et al. (2017).

---

## 10. Reproducing the Paper Tables

### Table 3 and 4 (nRMSE by timestep)

Run the relevant model script.  The per-timestep nRMSE table is printed to the terminal automatically.

### Table 5 (regularisation parameter grid)

In `models/model1_single_plume.py`, wrap the four inversion calls in loops over LAMBDA_S ∈ {10, 20, 40} and LAMBDA_T ∈ {10, 20, 40}.  Re-run and collect the `np.nanmean(errors[mn]['nrmse'])` value for each combination.

### Tables 6–8 (noise/error, alpha_max, epsilon sweeps)

Change `NOISE_LEVEL`, `DATA_ERROR`, `alpha_max`, or `epsilon` at the top of the script and re-run.  A convenience approach is to drive these from command-line arguments; this is left as a straightforward extension.

---

## 11. Adapting HWSI to New Datasets

The `HWSI_Inversion` class has no hard dependency on the synthetic model infrastructure.  To use it with real or differently structured data:

1. **Prepare data containers**: create one `DataContainerERT` per epoch.  Each container must have `rhoa` and `err` arrays.  Use pyGIMLi's `load` function for field data in standard formats.

2. **Build the inversion mesh**: construct a pyGIMLi mesh covering the survey area at an appropriate resolution.  A maximum cell area of 0.1–0.2 m² works well for near-surface surveys with electrode spacings around 0.5 m.

3. **Instantiate and run**:
```python
hwsi = HWSI_Inversion(inv_mesh, data_list, lambda_s=20, lambda_t=20)
results = hwsi.run()
```

4. **Inspect convergence**: check `hwsi.convergence_history` to verify that the mean update norm reached a stable plateau.  If the norm is still declining at `max_iter`, increase the iteration limit.

5. **Visualise**: use `interp_to_grid` to map per-cell results onto a regular grid for plotting.

Note that the performance of HWSI relative to the baselines in field conditions may differ from the synthetic results, as geological heterogeneity, variable electrode contact resistance, and non-stationary background shifts are not modelled in the current tests.

---

## 12. Common Issues and Fixes

**pyGIMLi version mismatch**  
The `TimelapseERT.fullInversion` API changed between minor versions.  The `run_4d` function handles this with a try/except block.  If you encounter an error, check that pyGIMLi is version 1.5.5.

**Very slow convergence**  
If the mean update norm does not plateau within 15 iterations, try increasing `lambda_s` slightly (e.g., from 20 to 30) to make individual spatial inversions better conditioned, or increase `max_iter`.

**HWSI worse than Independent at the first epoch (t = 4 h)**  
This is expected behaviour.  Only one prior model epoch is available at t = 4 h, so the HWSI weights default to unity and the method provides no advantage over uniform coupling.  The adaptive benefit accumulates from the third epoch onward once a two-step history is in place.

**NaN values in error output**  
This typically indicates that the interpolated model is entirely NaN at a particular epoch, meaning the inversion failed.  Check whether pyGIMLi printed any warnings or errors during that epoch's inversion call.

**Memory usage**  
Each HWSI iteration stores two model estimates per epoch (current and previous iteration).  For large meshes or long monitoring sequences, peak memory is proportional to 2 × n_cells × n_epochs × 8 bytes.  This is manageable for typical 2-D surveys.

---

*For further questions, please open a GitHub issue in the repository.*

---

## Author
**Garvit Gupta**

---