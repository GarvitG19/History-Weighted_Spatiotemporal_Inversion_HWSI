# HWSI – History-Weighted Spatiotemporal Inversion

**Cell-Adaptive Temporal Regularisation Framework for Time-Lapse Electrical Resistivity Tomography**

---

## Author

**Garvit Gupta**

---

## Overview

This repository contains the full Python implementation of HWSI (History-Weighted Spatiotemporal Inversion), along with the three benchmark inversion methods (Independent, Difference, and 4D L2-Coupled), synthetic data generation scripts for both test scenarios, and all figure-generation code used in the associated paper.

HWSI resolves a fundamental tension in time-lapse ERT inversion: the temporal coupling strength needed to suppress noise in stable regions is precisely the strength that blunts sensitivity to genuine subsurface change elsewhere. Rather than applying a single global temporal penalty, HWSI assigns each mesh cell its own coupling coefficient derived from the two most recently accepted model updates. Cells showing little change across consecutive epochs are anchored strongly to their temporal prior; cells undergoing rapid change are left free to respond to the current data. The weight field is refreshed at every Gauss-Seidel iteration and no preliminary estimation stage is required.

---

## Repository Structure

```
hwsi_repo/
│
├── LICENSE                          # MIT licence
├── README.md                        # This file
├── environment.yml                  # Conda environment (recommended)
├── requirements.txt                 # Python dependencies (pip fallback)
│
├── hwsi/
│   └── hwsi.py                      # Core HWSI class and all baseline methods
│
├── models/
│   ├── model1_single_plume.py       # Model 1: single expanding conductive plume
│   └── model2_two_plume.py          # Model 2: dual-plume scenario
│
├── notebooks/
│   ├── 01_model1_tutorial.ipynb     # Step-by-step tutorial for Model 1
│   └── 02_model2_tutorial.ipynb     # Step-by-step tutorial for Model 2
│
└── docs/
    └── user_guide.md                # Full user guide (inputs, outputs, options)
```

---

## Dependencies and System Requirements

| Package | Version tested | Purpose |
|---|---|---|
| Python | 3.9+ | Runtime |
| pyGIMLi | ≥ 1.5 | ERT forward modelling and inversion |
| NumPy | ≥ 1.23 | Numerical arrays |
| SciPy | ≥ 1.9 | Sparse matrices, interpolation |
| Matplotlib | ≥ 3.6 | Figures and plots |

No GPU is required. All experiments were run on a standard workstation CPU.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hwsi.git
cd hwsi
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate hwsi_env
```

This installs all dependencies including pyGIMLi automatically. No separate pip step is needed.

> **Note:** pyGIMLi links against compiled C++ libraries and must be installed via conda. The `environment.yml` handles this correctly. If you run into issues, refer to the [official pyGIMLi installation guide](https://www.pygimli.org/installation.html).

### 3. Verify the installation

```bash
python -c "import pygimli; print(pygimli.__version__)"
```

You should see a version number printed without any errors.

---

## Quick Start

### Reproducing Model 1 results (single plume)

```bash
python models/model1_single_plume.py
```

This script will:
1. Build the forward and inversion meshes.
2. Generate synthetic time-lapse ERT datasets for seven epochs (t = 0 to 24 h, every 4 h).
3. Run all four inversion methods: Independent, Difference, 4D L2-Coupled, and HWSI.
4. Print nRMSE tables to the terminal.
5. Save six TIFF figures to the working directory.

### Reproducing Model 2 results (two-plume scenario)

```bash
python models/model2_two_plume.py
```

Same workflow as above for the dual-plume scenario.

### Interactive step-by-step tutorials

Open the notebooks in Jupyter:

```bash
jupyter notebook notebooks/01_model1_tutorial.ipynb
jupyter notebook notebooks/02_model2_tutorial.ipynb
```

The notebooks walk through every stage — mesh setup, true-model construction, data generation, inversion, error evaluation, and figure generation — with inline explanations.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `lambda_s` | 20.0 | Spatial regularisation strength |
| `lambda_t` | 20.0 | Temporal coupling base strength |
| `epsilon` | 5e-3 | Singularity guard in weight denominator (log-resistivity units) |
| `alpha_max` | 0.65 | Hard cap on per-cell blending coefficient |
| `max_iter` | 15 | Maximum Gauss-Seidel iterations |

See `docs/user_guide.md` for a full description of all inputs, outputs, and options.

---

## Expected Outputs

Running either model script produces the following files in the working directory:

| File | Description |
|---|---|
| `model{1,2}_hwsi_true_models.tiff` | True resistivity at each monitoring epoch |
| `model{1,2}_hwsi_inversion_models.tiff` | Inverted resistivity for all four methods |
| `model{1,2}_hwsi_error_analysis.tiff` | nRMSE trajectories across time |
| `model{1,2}_hwsi_true_temporal_changes.tiff` | True log-ratio change maps |
| `model{1,2}_hwsi_inversion_temporal_changes.tiff` | Inverted change maps for all methods |
| `model{1,2}_hwsi_convergence.tiff` | HWSI convergence curve |

Terminal output includes a per-timestep nRMSE table and mean/std summary for each method.

---

## Reproducing All Paper Results

To reproduce all figures and tables from the paper in one go:

```bash
python models/model1_single_plume.py
python models/model2_two_plume.py
```

All datasets are generated programmatically from the scripts; no external data files are needed.

---

## Citing This Work

Developed by Garvit Gupta.
If you use this code in your research, please cite the associated paper (full citation to be added after publication).

---

## Contact

For questions about the code or methodology, please open a GitHub issue.

---

## Acknowledgements

The authors thank the developers of [pyGIMLi](https://www.pygimli.org/) (Rücker et al., 2017) for maintaining an open-source geophysical modelling and inversion framework that made this work possible.
