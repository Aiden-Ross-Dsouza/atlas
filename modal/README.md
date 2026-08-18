# Modal — ATLAS Contributor Guide

## Prerequisites

```bash
pip install modal
modal token new   # authenticate with your Modal account
```

## First run: create the persistent volume

```bash
modal volume create atlas-data
```

This volume persists all checkpoints, data, logs, and outputs across runs.

## Download data into the volume

```bash
modal run modal/modal_app.py::download_data
```

This runs `scripts/download_data.py` inside the container, saving everything to
the `atlas-data` volume at `/atlas_root`.

## Running experiments

```bash
# E0: adapter capacity (~3 GPU-h on A10G)
modal run modal/modal_app.py::run_e0

# E1: routing gate (~4 GPU-h on A10G)
modal run modal/modal_app.py::run_e1

# E4+E3: continual stream (~21 GPU-h on A10G)
modal run modal/modal_app.py::run_e4

# E2: 2×2 appearance vs dynamics (~6 GPU-h on A10G)
modal run modal/modal_app.py::run_e2
```

## Downloading results

```bash
modal volume get atlas-data /atlas_root/atlas_out ./atlas_out_from_modal
```

## GPU selection

The default GPU is `A10G`. To use a different GPU:

```python
@app.function(gpu="H100", ...)   # edit in modal_app.py
```

## Image updates

The Modal image installs from the published GitHub repo. After pushing changes:

```bash
modal deploy modal/modal_app.py   # rebuild and deploy
```

## Env vars inside the container

All ATLAS paths are pre-set via the image `.env()` call in `modal_app.py`:

| Variable        | Value in container               |
|-----------------|----------------------------------|
| `ATLAS_HOME`    | `/atlas_root`                    |
| `JEPAWM_DSET`   | `/atlas_root/data`               |
| `JEPAWM_CKPT`   | `/atlas_root/ckpts`              |
| `ATLAS_OUT`     | `/atlas_root/atlas_out`          |
| `TORCH_HOME`    | `/atlas_root/hub`                |
| `MUJOCO_GL`     | `egl`                            |
