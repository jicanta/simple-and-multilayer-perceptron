# 1-Baseline

- Mode: `baseline (no weighting)`
- Best epoch: `62`
- Elapsed: `480.2s`
- Test accuracy: `97.76%`
- Test F1 macro: `0.9773`
- Validation accuracy: `98.61%`
- Validation F1 macro: `0.9800`

## Hyperparameters

- `architecture`: `[784, 256, 128, 10]`
- `learning_rate`: `0.001`
- `optimizer`: `adam`
- `batch_size`: `128`
- `hidden_activation`: `leaky_relu`
- `output_activation`: `logistic`
- `leaky_relu_slope`: `0.01`
- `l2_lambda`: `0.0001`
- `epochs`: `180`
- `patience`: `24`
- `normalization`: `minmax`
- `augment_repeats`: `1`
- `augment_shift_max`: `2`
- `augment_noise_std`: `0.03`

## Test Highlights

Top classes by test F1:
- class `1`: F1 `0.9947`, acc/recall `99.65%`
- class `0`: F1 `0.9878`, acc/recall `99.18%`
- class `4`: F1 `0.9857`, acc/recall `98.37%`

Weakest classes by test F1:
- class `9`: F1 `0.9584`, acc/recall `96.03%`
- class `5`: F1 `0.9664`, acc/recall `96.86%`
- class `8`: F1 `0.9688`, acc/recall `95.88%`

## Included Files

- `hyperparameters.json`
- `metrics_summary.json`
- `full_result.json`
- `training_curves.png`
- `confusion_matrix.png`
- `saliency_map.png`
- `first_layer_weights.png`
