# Coagulation–Fragmentation Control (JAX)

JAX code for gradient-based optimal control experiments for a coagulation–fragmentation PDE.

> **Status**: research / proof-of-concept code (focused on qualitative behavior, not high-order numerics).

## What’s inside

This repository contains:
- a forward solver for a coagulation–fragmentation model (`solver.py`)
- an adjoint solver (`adjoint_solver.py`)
- a (gradient-based) optimization routine with PMP check (`optimizer.py`)
- plotting helpers (`plotting_utils.py`)
- small data structures / containers (`data_classes.py`)
- a runnable notebook entry point (`main.ipynb`)

## Quick start

### 1) Create a virtual environment

    python -m venv .venv
    # Linux/macOS:
    source .venv/bin/activate
    # Windows (PowerShell):
    # .venv\Scripts\Activate.ps1
    python -m pip install -U pip

### 2) Install dependencies

    pip install -r requirements.txt

### 3) Run the experiments

    jupyter lab

Then open `main.ipynb` and run the cells.

## Reproducibility notes

- This is research code: results may depend on the JAX version and hardware.

## How to cite

Repository:
- https://github.com/Sart0/coagulation-fragmentation-control

For paper-grade reproducibility, cite a tagged release or a specific commit hash.


