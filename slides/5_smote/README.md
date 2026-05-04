# 5-SMOTE

- Mode: `smote`
- Best epoch: `66`
- Elapsed: `525.2s`
- Test accuracy: `97.96%`
- Test F1 macro: `0.9793`
- Validation accuracy: `98.49%`
- Validation F1 macro: `0.9751`

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
- `synthetic_balancing`: `False`
- `synthetic_shift_max`: `1`
- `synthetic_noise_std`: `0.02`
- `synthetic_mix_min`: `0.35`
- `synthetic_mix_max`: `0.65`
- `smote`: `True`
- `smote_k_neighbors`: `5`

## Test Highlights

Top classes by test F1:
- class `4`: F1 `0.9980`, acc/recall `99.59%`
- class `1`: F1 `0.9912`, acc/recall `99.29%`
- class `0`: F1 `0.9839`, acc/recall `99.59%`

Weakest classes by test F1:
- class `8`: F1 `0.9617`, acc/recall `93.00%`
- class `3`: F1 `0.9689`, acc/recall `98.81%`
- class `5`: F1 `0.9709`, acc/recall `97.31%`

## Included Files

- `hyperparameters.json`
- `metrics_summary.json`
- `full_result.json`
- `training_curves.png`
- `confusion_matrix.png`
- `saliency_map.png`
- `first_layer_weights.png`
