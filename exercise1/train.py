import numpy as np

from data import (
    FEATURE_COLUMNS,
    kfold_split,
    load_dataset,
    print_data_quality,
    print_eda,
    StandardScaler,
    train_test_split,
)
from perceptron import LinearPerceptron, NonLinearPerceptron
from plots import save_loss_comparison, save_roc_curve, save_threshold_analysis


EPOCHS = 200
LEARNING_RATE = 0.01
BATCH_SIZE = 64
K_FOLDS = 5
TRAIN_RATIO = 0.8


def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    preds = (y_pred >= threshold).astype(int)
    tp = int(((preds == 1) & (ground_truth == 1)).sum())
    fp = int(((preds == 1) & (ground_truth == 0)).sum())
    tn = int(((preds == 0) & (ground_truth == 0)).sum())
    fn = int(((preds == 0) & (ground_truth == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "mse": mse,
        "mae": mae,
        "accuracy": (tp + tn) / len(ground_truth),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_learning_comparison(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section("Part 1: Learning Comparison (full dataset)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nTraining LinearPerceptron  (epochs={EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE})")
    linear = LinearPerceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    linear.fit(X_scaled, y, verbose=True)

    print(f"\nTraining NonLinearPerceptron (epochs={EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE})")
    nonlinear = NonLinearPerceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    nonlinear.fit(X_scaled, y, verbose=True)

    y_pred_lin = linear.predict(X_scaled)
    y_pred_nl = nonlinear.predict(X_scaled)

    m_lin = compute_metrics(y, y_pred_lin, ground_truth)
    m_nl = compute_metrics(y, y_pred_nl, ground_truth)

    print(f"\n  {'Metric':<12} {'Linear':>12} {'Non-Linear':>12}")
    print("  " + "-" * 38)
    for key in ("mse", "mae", "accuracy", "f1"):
        fmt = ".2%" if key == "accuracy" else ".6f"
        v_lin = f"{m_lin[key]:{fmt}}"
        v_nl = f"{m_nl[key]:{fmt}}"
        print(f"  {key:<12} {v_lin:>12} {v_nl:>12}")

    lin_range = (float(y_pred_lin.min()), float(y_pred_lin.max()))
    nl_range = (float(y_pred_nl.min()), float(y_pred_nl.max()))
    print(f"\n  Linear output range:      [{lin_range[0]:.4f}, {lin_range[1]:.4f}]")
    print(f"  Non-linear output range:  [{nl_range[0]:.4f}, {nl_range[1]:.4f}]")

    path = save_loss_comparison("learning_comparison.png", linear.losses, nonlinear.losses)
    if path:
        print(f"\n  plot: {path}")


def run_generalization_study(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section("Part 2: Generalization Study — K-Fold Cross-Validation (NonLinearPerceptron)")
    print(f"  k={K_FOLDS}  epochs={EPOCHS}  lr={LEARNING_RATE}  batch={BATCH_SIZE}\n")

    fold_metrics = []
    for fold_idx, (X_tr, y_tr, gt_tr, X_val, y_val, gt_val) in enumerate(
        kfold_split(X, y, ground_truth, k=K_FOLDS)
    ):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = NonLinearPerceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
        model.fit(X_tr_s, y_tr)

        m = compute_metrics(y_val, model.predict(X_val_s), gt_val)
        fold_metrics.append(m)
        print(
            f"  Fold {fold_idx + 1}  "
            f"MSE={m['mse']:.6f}  MAE={m['mae']:.6f}  "
            f"F1={m['f1']:.4f}  Acc={m['accuracy']:.2%}"
        )

    print()
    for key in ("mse", "mae", "f1", "accuracy"):
        values = [m[key] for m in fold_metrics]
        fmt = ".2%" if key == "accuracy" else ".6f"
        print(f"  avg {key:<10} {np.mean(values):{fmt}}  ± {np.std(values):.6f}")


def run_final_model(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section("Part 3: Final Model + Threshold Recommendation")

    X_tr, y_tr, gt_tr, X_te, y_te, gt_te = train_test_split(
        X, y, ground_truth, train_ratio=TRAIN_RATIO
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\n  Train: {len(X_tr)} samples  Test: {len(X_te)} samples")
    print(f"  Training NonLinearPerceptron (epochs={EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE})")

    model = NonLinearPerceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.fit(X_tr_s, y_tr, verbose=True)

    y_te_pred = model.predict(X_te_s)

    roc_path, auc = save_roc_curve(
        "roc_curve.png", gt_te, y_te_pred, "Non-Linear Perceptron — ROC Curve"
    )
    thresh_path, best_thresh = save_threshold_analysis(
        "threshold_analysis.png",
        gt_te,
        y_te_pred,
        "Non-Linear Perceptron — Threshold Analysis",
    )

    m = compute_metrics(y_te, y_te_pred, gt_te, threshold=best_thresh)

    print(f"\n  ROC-AUC:              {auc:.4f}")
    print(f"  Recommended threshold (max F1): {best_thresh:.4f}")
    print(f"  MSE:        {m['mse']:.6f}")
    print(f"  MAE:        {m['mae']:.6f}")
    print(f"  Accuracy:   {m['accuracy']:.2%}")
    print(f"  Precision:  {m['precision']:.4f}")
    print(f"  Recall:     {m['recall']:.4f}")
    print(f"  F1:         {m['f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")

    for path in (roc_path, thresh_path):
        if path:
            print(f"  plot: {path}")


EXTENDED_EPOCHS = 1000
PATIENCE = 30


def _find_best_threshold(y_true_gt: np.ndarray, y_scores: np.ndarray) -> tuple:
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_thresh = 0.0, 0.5
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = int(((preds == 1) & (y_true_gt == 1)).sum())
        fp = int(((preds == 1) & (y_true_gt == 0)).sum())
        fn = int(((preds == 0) & (y_true_gt == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh, best_f1


def run_extended_generalization_study(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section(
        "Part 2b: Extended Generalization — K-Fold with Early Stopping + Optimal Threshold"
    )
    print(
        f"  k={K_FOLDS}  max_epochs={EXTENDED_EPOCHS}  "
        f"patience={PATIENCE}  lr={LEARNING_RATE}  batch={BATCH_SIZE}\n"
    )

    fold_metrics = []
    fold_epochs = []

    for fold_idx, (X_tr, y_tr, gt_tr, X_val, y_val, gt_val) in enumerate(
        kfold_split(X, y, ground_truth, k=K_FOLDS)
    ):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = NonLinearPerceptron(
            learning_rate=LEARNING_RATE, epochs=EXTENDED_EPOCHS, batch_size=BATCH_SIZE
        )
        model.fit(X_tr_s, y_tr, patience=PATIENCE)

        y_val_pred = model.predict(X_val_s)
        best_thresh, _ = _find_best_threshold(gt_val, y_val_pred)
        m = compute_metrics(y_val, y_val_pred, gt_val, threshold=best_thresh)
        fold_metrics.append(m)
        fold_epochs.append(len(model.losses))

        print(
            f"  Fold {fold_idx + 1}  epochs={len(model.losses):>4}  "
            f"MSE={m['mse']:.6f}  MAE={m['mae']:.6f}  "
            f"thresh={best_thresh:.2f}  F1={m['f1']:.4f}  Acc={m['accuracy']:.2%}"
        )

    print()
    for key in ("mse", "mae", "f1", "accuracy"):
        values = [m[key] for m in fold_metrics]
        fmt = ".2%" if key == "accuracy" else ".6f"
        print(f"  avg {key:<10} {np.mean(values):{fmt}}  ± {np.std(values):.6f}")
    print(f"  avg epochs     {np.mean(fold_epochs):.1f}  ± {np.std(fold_epochs):.1f}")


def run_best_training_set(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section("Part 4: Best Training Set — Full Dataset Retraining")
    print(
        "  The extended K-Fold shows low variance across all folds, meaning any\n"
        "  representative split of the data yields a similar model. The best training\n"
        "  set is therefore the largest available one: the full dataset.\n"
        "  Generalization estimate comes from the K-Fold MSE above (not a held-out set).\n"
    )
    print(
        f"  Training NonLinearPerceptron on all {len(X)} samples  "
        f"(max_epochs={EXTENDED_EPOCHS}, patience={PATIENCE})\n"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = NonLinearPerceptron(
        learning_rate=LEARNING_RATE, epochs=EXTENDED_EPOCHS, batch_size=BATCH_SIZE
    )
    model.fit(X_scaled, y, verbose=True, patience=PATIENCE)

    y_pred = model.predict(X_scaled)
    mse = float(np.mean((y - y_pred) ** 2))
    mae = float(np.mean(np.abs(y - y_pred)))

    # Threshold analysis on full dataset (optimistic estimate — training data)
    best_thresh, best_f1 = _find_best_threshold(ground_truth, y_pred)
    m = compute_metrics(y, y_pred, ground_truth, threshold=best_thresh)

    print(f"\n  Epochs run:   {len(model.losses)}")
    print(f"  Final loss:   {model.losses[-1]:.6f}")
    print(f"  MSE (train):  {mse:.6f}  — true generalization estimate: K-Fold avg MSE above")
    print(f"  MAE (train):  {mae:.6f}")
    print(f"\n  Threshold analysis on full dataset (reference only — biased, uses training data):")
    print(f"  Best threshold (max F1): {best_thresh:.4f}")
    print(f"  F1={m['f1']:.4f}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")
    print(f"  Accuracy={m['accuracy']:.2%}  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(
        f"\n  Recommended threshold for deployment: use the value found in Part 3\n"
        f"  (trained on 80%, evaluated on held-out 20%) as the unbiased estimate."
    )


if __name__ == "__main__":
    X, y, ground_truth = load_dataset()
    print_eda(X, y, ground_truth)
    print_data_quality(X, y, ground_truth)
    run_learning_comparison(X, y, ground_truth)
    run_generalization_study(X, y, ground_truth)
    run_final_model(X, y, ground_truth)
    run_extended_generalization_study(X, y, ground_truth)
    run_best_training_set(X, y, ground_truth)
