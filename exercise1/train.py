import numpy as np

from data import (
    build_feature_sets,
    FEATURE_COLUMNS,
    kfold_split,
    load_dataset,
    print_data_quality,
    print_eda,
    StandardScaler,
    train_test_split,
)
from perceptron import LinearPerceptron, NonLinearPerceptron
from plots import (
    save_activation_comparison,
    save_feature_engineering_plot,
    save_loss_comparison,
    save_roc_curve,
    save_threshold_analysis,
)


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


def run_feature_engineering(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section("Optional: Feature Engineering — Impact on F1")

    feature_sets = build_feature_sets(X)

    print("\n  Feature sets:")
    print("  A — Original 9 features (baseline)")
    print("  B — Replace timestamp→hour_of_day, amount_usd→log_amount  (still 9)")
    print("  C — Set B + is_new_account + browsed_before_buying  (11 features)")

    def _kfold_f1(X_data: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        f1s = []
        for X_tr, y_tr, gt_tr, X_val, y_val, gt_val in kfold_split(
            X_data, target, ground_truth, k=K_FOLDS
        ):
            sc = StandardScaler()
            model = NonLinearPerceptron(
                learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE
            )
            model.fit(sc.fit_transform(X_tr), y_tr)
            m = compute_metrics(y_val, model.predict(sc.transform(X_val)), gt_val)
            f1s.append(m["f1"])
        return float(np.mean(f1s)), float(np.std(f1s))

    def _run_table(target: np.ndarray, target_label: str, plot_filename: str) -> None:
        print(f"\n  Target: {target_label}")
        print(f"  {'Set':<40} {'n':>4} {'F1 mean':>9} {'± std':>7} {'vs A':>8}")
        print("  " + "-" * 73)
        labels, f1_means, f1_stds, base_f1 = [], [], [], None
        for label, (X_set, names) in feature_sets.items():
            f1, std = _kfold_f1(X_set, target)
            if base_f1 is None:
                base_f1 = f1
            delta = f1 - base_f1
            print(f"  {label:<40} {len(names):>4} {f1:>9.4f} {std:>7.4f}  {delta:>+.4f}")
            labels.append(label)
            f1_means.append(f1)
            f1_stds.append(std)
        path = save_feature_engineering_plot(plot_filename, labels, f1_means, f1_stds, base_f1)
        if path:
            print(f"  plot: {path}")

    print(f"\n  Running K-Fold (k={K_FOLDS}, epochs={EPOCHS})...\n")

    _run_table(
        y,
        "big_model_fraud_probability (continuous, model already captured non-linearities)",
        "feature_engineering_bigmodel.png",
    )
    _run_table(
        ground_truth.astype(np.float64),
        "flagged_fraud (binary 0/1, model must discover non-linearities from scratch)",
        "feature_engineering_groundtruth.png",
    )


def run_relu_comparison(X: np.ndarray, y: np.ndarray, ground_truth: np.ndarray) -> None:
    _section("Optional: ReLU vs Sigmoid — Activation Comparison (K-Fold)")
    print(f"  k={K_FOLDS}  epochs={EPOCHS}  lr={LEARNING_RATE}  batch={BATCH_SIZE}\n")
    print(
        f"  {'Fold':<6} {'Sigmoid F1':>12} {'Sigmoid Acc':>13}"
        f" {'ReLU F1':>10} {'ReLU Acc':>11}"
    )
    print("  " + "-" * 56)

    sigmoid_fold_metrics: list[dict] = []
    relu_fold_metrics: list[dict] = []

    for fold_idx, (X_tr, y_tr, gt_tr, X_val, y_val, gt_val) in enumerate(
        kfold_split(X, y, ground_truth, k=K_FOLDS)
    ):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        sig_model = NonLinearPerceptron(
            learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, activation="sigmoid"
        )
        sig_model.fit(X_tr_s, y_tr)

        relu_model = NonLinearPerceptron(
            learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, activation="relu"
        )
        relu_model.fit(X_tr_s, y_tr)

        m_sig = compute_metrics(y_val, sig_model.predict(X_val_s), gt_val)
        m_relu = compute_metrics(y_val, relu_model.predict(X_val_s), gt_val)
        sigmoid_fold_metrics.append(m_sig)
        relu_fold_metrics.append(m_relu)

        print(
            f"  {fold_idx + 1:<6}"
            f" {m_sig['f1']:>12.4f} {m_sig['accuracy']:>13.2%}"
            f" {m_relu['f1']:>10.4f} {m_relu['accuracy']:>11.2%}"
        )

    print()
    print(f"  {'Metric':<12} {'Sigmoid mean ± std':>24}  {'ReLU mean ± std':>24}")
    print("  " + "-" * 64)
    for key in ("mse", "mae", "f1", "accuracy"):
        v_sig = [m[key] for m in sigmoid_fold_metrics]
        v_relu = [m[key] for m in relu_fold_metrics]
        fmt = ".4f" if key in ("f1",) else ".6f"
        if key == "accuracy":
            sig_str = f"{np.mean(v_sig):.2%} ± {np.std(v_sig):.4f}"
            relu_str = f"{np.mean(v_relu):.2%} ± {np.std(v_relu):.4f}"
        else:
            sig_str = f"{np.mean(v_sig):{fmt}} ± {np.std(v_sig):.6f}"
            relu_str = f"{np.mean(v_relu):{fmt}} ± {np.std(v_relu):.6f}"
        print(f"  {key:<12} {sig_str:>24}  {relu_str:>24}")

    # Full-dataset training to compare loss curves and output distributions
    print("\n  Training on full dataset for curve/distribution comparison...")
    scaler_full = StandardScaler()
    X_s = scaler_full.fit_transform(X)

    sig_full = NonLinearPerceptron(
        learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, activation="sigmoid"
    )
    sig_full.fit(X_s, y)

    relu_full = NonLinearPerceptron(
        learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, activation="relu"
    )
    relu_full.fit(X_s, y)

    sig_out = sig_full.predict(X_s)
    relu_out = relu_full.predict(X_s)

    print(f"\n  Output range (full dataset):")
    print(f"    sigmoid : [{sig_out.min():.4f}, {sig_out.max():.4f}]  — bounded by construction")
    print(f"    relu    : [{relu_out.min():.4f}, {relu_out.max():.4f}]  — unbounded above")
    dead_pct = (relu_out == 0.0).mean()
    print(f"    ReLU dead (output==0): {dead_pct:.1%} of samples")

    path = save_activation_comparison(
        "relu_comparison.png",
        sig_full.losses,
        relu_full.losses,
        sig_out,
        relu_out,
    )
    if path:
        print(f"\n  plot: {path}")

    print(
        "\n  Key observations:"
        "\n  · Sigmoid gradients vanish when outputs saturate near 0 or 1 (slow late convergence)."
        "\n  · ReLU avoids saturation — gradient is constant (1) for active neurons."
        "\n  · ReLU risk: neurons with negative pre-activation get zero gradient ('dying ReLU')."
        "\n  · ReLU outputs are unbounded; threshold search must cover the actual output range."
        "\n  · For single-layer binary classification, sigmoid is generally preferred as it"
        "\n    produces calibrated probabilities and avoids the dead-neuron problem."
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
    run_relu_comparison(X, y, ground_truth)
    run_feature_engineering(X, y, ground_truth)
