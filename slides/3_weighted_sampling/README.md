# 3-Weighted-Sampling

- Mode: `weighted sampling`
- Best epoch: `73`
- Elapsed: `487.8s`
- Test accuracy: `98.28%`
- Test F1 macro: `0.9825`
- Validation accuracy: `98.54%`
- Validation F1 macro: `0.9792`

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
- class `1`: F1 `0.9947`, acc/recall `100.00%`
- class `4`: F1 `0.9898`, acc/recall `98.78%`
- class `3`: F1 `0.9880`, acc/recall `98.41%`

Weakest classes by test F1:
- class `9`: F1 `0.9743`, acc/recall `97.62%`
- class `8`: F1 `0.9748`, acc/recall `95.47%`
- class `5`: F1 `0.9755`, acc/recall `98.21%`

## Included Files

- `hyperparameters.json`
- `metrics_summary.json`
- `full_result.json`
- `training_curves.png`
- `confusion_matrix.png`
- `saliency_map.png`
- `first_layer_weights.png`
