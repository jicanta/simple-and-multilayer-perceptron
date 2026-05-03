# Exercise 1 Visualizations

This directory keeps the dedicated visualization artifacts for Exercise 1 separate from the generic metric plots in `exercise1/plots/`.

Structure:

- `data/`: JSON exported by `python3 exercise1/train.py` for the final model.
- `manim/`: Manim scene code that renders the main network overview.
- `renders/`: Manim output directory.

Workflow:

1. Generate the latest final-model data:

```bash
python3 exercise1/train.py
```

2. Render the Manim overview scene:

```bash
bash exercise1/visualizations/manim/render_final_model.sh
```

Notes:

- The Manim scene reads `exercise1/visualizations/data/final_model_overview.json`.
- The scene includes the network structure, learned weights, split sizes, metrics, threshold, and training hyperparameters.
- The render script uses the official Docker image `manimcommunity/manim:stable`, so no local Manim install is required.
