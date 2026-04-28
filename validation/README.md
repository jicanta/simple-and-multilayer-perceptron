# Validation

This folder contains small, self-contained validation scripts for the perceptrons requested in the TP.

Covered validations:

- `single-layer-step-perceptron.py`: logical `AND` with outputs in `{-1, 1}`.
- `single-layer-linear-perceptron.py`: linear fits over several target functions such as `y = x`, `y = 2x`, `y = x + 1`, and `y = -1.5x + 0.5`.
- `single-layer-non-linear-perceptron.py`: non-linear fit over samples from `y = tanh(x)`.
- `multi-layer-perceptron.py`: handcrafted multilayer solutions for logical `XOR` using architectures `[2, 2, 1]` and `[2, 3, 2, 1]`.

## What was added

- Standardized console reports across all scripts.
- Reusable training and evaluation functions so the validations can be tested.
- Extra assertions in `test_validation_scripts.py`.
- Optional visualizations saved under `validation/plots/` when `matplotlib` is available.
- Held-out evaluation for the regression validations to avoid training/evaluation leakage.

## How to run

From the repository root:

```bash
python3 validation/single-layer-step-perceptron.py
python3 validation/single-layer-linear-perceptron.py
python3 validation/single-layer-non-linear-perceptron.py
python3 validation/multi-layer-perceptron.py
```

To run all automated checks:

```bash
python3 -m unittest validation/test_validation_scripts.py
```

## Visualizations

If `matplotlib` is installed, the scripts generate plots automatically in `validation/plots/`.

Generated plots include:

- loss curves for the trainable perceptrons
- regression overlays for the linear and non-linear fits
- residual plots on held-out samples for the regression validations
- expected-vs-predicted scatter plots for the regression validations
- expected vs predicted point layouts for the logical validations

If `matplotlib` is not installed, the scripts still run and only skip plot generation.

## What to look at

- Step perceptron: zero classification errors on `AND`.
- Linear perceptron: low held-out MSE and fitted parameters close to the target function.
- Non-linear perceptron: low held-out MSE and a fitted curve close to `tanh(x)`.
- Multilayer perceptron: exact `XOR` classification for both requested architectures.

## Leakage note

The logical `AND` and `XOR` scripts are deterministic sanity checks over tiny truth tables, so they intentionally evaluate the same full set they define.

The regression validations now split the samples into train and held-out test partitions before training. That keeps the reported regression metrics free from train/test leakage while still preserving the simplicity of the TP validation exercises.
