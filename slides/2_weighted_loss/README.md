# 2-Weighted-Loss

- Mode: `weighted loss`
- Best epoch: `74`
- Elapsed: `529.4s`
- Test accuracy: `98.16%`
- Test F1 macro: `0.9814`
- Validation accuracy: `98.63%`
- Validation F1 macro: `0.9803`

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
- class `1`: F1 `0.9982`, acc/recall `99.65%`
- class `3`: F1 `0.9921`, acc/recall `100.00%`
- class `4`: F1 `0.9899`, acc/recall `100.00%`

Weakest classes by test F1:
- class `7`: F1 `0.9653`, acc/recall `97.28%`
- class `8`: F1 `0.9665`, acc/recall `95.06%`
- class `9`: F1 `0.9722`, acc/recall `97.22%`

## Included Files

- `hyperparameters.json`
- `metrics_summary.json`
- `full_result.json`
- `training_curves.png`
- `confusion_matrix.png`
- `saliency_map.png`
- `first_layer_weights.png`
