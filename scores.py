import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# ---------------------------
# Replace these with your data
# ---------------------------
# y_true: shape (n_samples,) with integer labels 0..n_classes-1
# preds:  shape (n_samples, n_classes) with probability-like scores (or decision scores)
# Example (toy) - remove/replace:
# np.random.seed(0)
# n_samples = 200
# n_classes = 4
# y_true = np.random.randint(0, n_classes, size=n_samples)
# preds_logits = np.random.randn(n_samples, n_classes)
# preds = np.exp(preds_logits) / np.exp(preds_logits).sum(axis=1, keepdims=True)

# Use real y_true and preds from your model
# y_true = ...
# preds = ...

# ---------------------------
# Sanity checks
# ---------------------------
def getScoresXXX(y_true, preds, wantToSave=False, pathToSave=None, fnToSave=None):
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    if preds.ndim != 2:
        raise ValueError("preds must be shape (n_samples, n_classes)")
    n_samples, n_classes = preds.shape
    if y_true.shape[0] != n_samples:
        raise ValueError("y_true and preds must have same number of samples")
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    y_pred_labels = np.argmax(preds, axis=1)

    # ---------------------------
    # Overall scalar metrics
    # ---------------------------
    overall_accuracy = accuracy_score(y_true, y_pred_labels)
    overall_macro_f1 = f1_score(y_true, y_pred_labels, average="macro")
    overall_mcc = matthews_corrcoef(y_true, y_pred_labels)
    pr_auc_macro = average_precision_score(y_true_bin, preds, average="macro")
    auc_macro = roc_auc_score(y_true_bin, preds, average="macro", multi_class="ovr")

    if not wantToSave:
        scores = {'acc': overall_accuracy,
                  'f1': overall_macro_f1,
                  'pr_auc': pr_auc_macro,
                  'roc_auc': auc_macro,
                  'mcc': overall_mcc
                  }
        return scores
    else:
        # ---------------------------
        # Per-class ROC AUC and ROC curves
        # ---------------------------
        per_class_auc = []
        roc_curve_dfs = []  # list of DataFrames (one per class) to concatenate
        for c in range(n_classes):
            fpr, tpr, thr = roc_curve(y_true_bin[:, c], preds[:, c])
            auc_c = roc_auc_score(y_true_bin[:, c], preds[:, c])
            per_class_auc.append(auc_c)

            df_c = pd.DataFrame({
                "class": [c] * len(fpr),
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thr
            })
            df_c["auc"] = auc_c  # constant per row for convenience
            roc_curve_dfs.append(df_c)

        # Micro-average ROC curve (flattened)
        fpr_micro, tpr_micro, thr_micro = roc_curve(y_true_bin.ravel(), preds.ravel())

        roc_micro_df = pd.DataFrame({
            "class": ["micro"] * len(fpr_micro),
            "fpr": fpr_micro,
            "tpr": tpr_micro,
            "threshold": thr_micro,
            "auc": [auc_macro] * len(fpr_micro)
        })

        # Concatenate all ROC curve points into one DataFrame
        roc_curves_df = pd.concat(roc_curve_dfs + [roc_micro_df], ignore_index=True)

        # ---------------------------
        # Prepare metrics summary DataFrame
        # ---------------------------
        rows = []
        class_counts = y_true_bin.sum(axis=0).astype(int)
        for c in range(n_classes):
            rows.append({
                "class": c,
                "support": int(class_counts[c]),
                "roc_auc": per_class_auc[c],
            })
        metrics_df = pd.DataFrame(rows).set_index("class")

        # Add overall aggregate row(s) as separate DataFrame (so CSV is easy to read)
        agg = {
            "name": ["overall_accuracy", "overall_macro_f1", "overall_mcc",
                     "pr_auc_macro", "roc_auc_macro"],
            "value": [overall_accuracy, overall_macro_f1, overall_mcc,
                      pr_auc_macro, auc_macro]
        }
        agg_df = pd.DataFrame(agg)

        # ---------------------------
        # Save to CSV
        # ---------------------------
        metrics_df.to_csv(os.path.join(pathToSave, fnToSave +"_per_class_metrics.csv"))       # per-class ROC AUC and support
        agg_df.to_csv(os.path.join(pathToSave, fnToSave +"overall_metrics.csv"), index=False) # overall scalar metrics
        roc_curves_df.to_csv(os.path.join(pathToSave, fnToSave +"roc_curves.csv"), index=False) # all ROC point data (per-class + micro)

        print("Saved:")
        print(" - per_class_metrics.csv")
        print(" - overall_metrics.csv")
        print(" - roc_curves.csv")

