# 4-Synthetic-Balancing

- Mode: `synthetic balancing (SMOTE-like)`
- Best epoch: `32`
- Elapsed: `283.8s`
- Test accuracy: `97.96%`
- Test F1 macro: `0.9791`
- Validation accuracy: `98.37%`
- Validation F1 macro: `0.9766`

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
- `synthetic_balancing`: `True`
- `synthetic_shift_max`: `1`
- `synthetic_noise_std`: `0.02`
- `synthetic_mix_min`: `0.35`
- `synthetic_mix_max`: `0.65`

## Test Highlights

Top classes by test F1:
- class `4`: F1 `0.9959`, acc/recall `100.00%`
- class `1`: F1 `0.9912`, acc/recall `99.29%`
- class `0`: F1 `0.9859`, acc/recall `99.59%`

Weakest classes by test F1:
- class `5`: F1 `0.9557`, acc/recall `91.93%`
- class `9`: F1 `0.9718`, acc/recall `95.63%`
- class `3`: F1 `0.9728`, acc/recall `99.21%`

## Included Files

- `hyperparameters.json`
- `metrics_summary.json`
- `full_result.json`
- `training_curves.png`
- `confusion_matrix.png`
- `saliency_map.png`
- `first_layer_weights.png`
