import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

from src.baseline import baseline_scores
from src.config import HEALTHY_RUL, METRICS_DIR, PROCESSED_DIR, ROOT
from src.evaluation import evaluate, lead_time_summary, per_engine_lead_times

FIG_DIR = ROOT / "results" / "figures"
Q_TARGET = 0.995   # threshold = this percentile of healthy validation scores, for BOTH methods
ZONE_BINS = [-1, 15, 30, 50, 75, 100, 1e9]
ZONE_LABELS = ["0-15", "16-30", "31-50", "51-75", "76-100", ">100"]
ZONE_ORDER = ZONE_LABELS[::-1]          # left to right = approaching failure
METHODS = ["Z-score baseline", "Isolation Forest"]


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def load_split(name, k):
    df = pd.read_csv(PROCESSED_DIR / f"features_{name}.csv")
    sc = pd.read_csv(PROCESSED_DIR / f"scores_{name}.csv")
    assert (df["unit"].to_numpy() == sc["unit"].to_numpy()).all()
    assert (df["cycle"].to_numpy() == sc["cycle"].to_numpy()).all()
    scores = {"Z-score baseline": baseline_scores(df, k),
              "Isolation Forest": sc["anomaly_score"].to_numpy()}
    return df, scores


def plot_roc_pr(data, ops):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, split in enumerate(("val", "test")):
        df, scores = data[split]
        y = df["is_anomaly"].to_numpy()
        for method in METHODS:
            s = scores[method]
            fpr, tpr, _ = roc_curve(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            line, = axes[i, 0].plot(fpr, tpr, label=method)
            axes[i, 1].plot(rec, prec, color=line.get_color(),
                            label=f"{method} (AP {average_precision_score(y, s):.2f})")
            m = ops[(split, method)]["metrics"]
            axes[i, 0].scatter(m["false_alarm_rate"], m["recall"], color=line.get_color(), marker="X", s=80)
            axes[i, 1].scatter(m["recall"], m["precision"], color=line.get_color(), marker="X", s=80)
        axes[i, 0].plot([0, 1], [0, 1], "k--", lw=0.7)
        axes[i, 0].set(title=f"ROC ({split}); X = operating point", xlabel="False alarm rate", ylabel="Recall")
        axes[i, 1].set(title=f"Precision-recall ({split})", xlabel="Recall", ylabel="Precision")
        axes[i, 0].legend(loc="lower right")
        axes[i, 1].legend(loc="lower left")
    fig.tight_layout()
    save_fig(fig, "05_roc_pr.png")


def plot_zone_rates(data, thr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        df, scores = data[split]
        zone = pd.cut(df["rul"], bins=ZONE_BINS, labels=ZONE_LABELS)
        rates = pd.DataFrame({
            m: pd.Series(scores[m] > thr[m]).groupby(zone.to_numpy(), observed=False).mean()
            for m in METHODS}).reindex(ZONE_ORDER)
        rates.plot(kind="bar", ax=ax, rot=0)
        ax.set(title=f"Share of rows flagged by cycles-before-failure zone ({split})",
               xlabel="Cycles before failure", ylabel="Alarm rate")
    fig.tight_layout()
    save_fig(fig, "06_alarm_rate_by_zone.png")


def plot_lead_times(data, thr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, split in zip(axes, ("val", "test")):
        df, scores = data[split]
        vals, labels = [], []
        for pos, m in enumerate(METHODS, start=1):
            leads = per_engine_lead_times(df, scores[m] > thr[m])
            found = leads.dropna().to_numpy()
            vals.append(found)
            labels.append(f"{m}\nmissed {int(leads.isna().sum())}/{len(leads)}")
            ax.scatter(pos + rng.uniform(-0.12, 0.12, len(found)), found, alpha=0.6, s=18)
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set(title=f"Detection lead time per engine ({split})", ylabel="Cycles before failure at first alarm")
    fig.tight_layout()
    save_fig(fig, "07_lead_time.png")


def plot_confusion(data, thr):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for i, split in enumerate(("val", "test")):
        df, scores = data[split]
        for j, m in enumerate(METHODS):
            r = evaluate(df, scores[m] > thr[m], scores[m])
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i, j],
                        xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
            axes[i, j].set(title=f"{m} ({split})", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    save_fig(fig, "08_confusion_matrices.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "baseline_metrics.json") as f:
        k = json.load(f)["tuned|val"]["k"]

    data = {s: load_split(s, k) for s in ("val", "test")}

    # label-free calibration: same percentile of healthy validation scores for both methods
    val_df, val_scores = data["val"]
    healthy = (val_df["rul"] > HEALTHY_RUL).to_numpy()
    thr = {m: float(np.quantile(val_scores[m][healthy], Q_TARGET)) for m in METHODS}

    ops, rows = {}, []
    for split in ("val", "test"):
        df, scores = data[split]
        for m in METHODS:
            flags = scores[m] > thr[m]
            met, lt = evaluate(df, flags, scores[m]), lead_time_summary(df, flags)
            ops[(split, m)] = {"metrics": met, "lead_time": lt}
            rows.append({"split": split, "method": m, "precision": met["precision"],
                         "recall": met["recall"], "f1": met["f1"], "auc": met["roc_auc"],
                         "far": met["false_alarm_rate"], "healthy_far": met["healthy_false_alarm_rate"],
                         "detected": f"{lt['detected']}/{lt['engines']}", "median_lead": lt["median_lead"]})

    table = pd.DataFrame(rows)
    table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(f"=== MATCHED OPERATING POINT: {Q_TARGET} percentile of healthy validation scores ===")
    print(table.round(3).to_string(index=False))

    print("\n=== ENGINES MISSED (no persistent alarm before failure) ===")
    for split in ("val", "test"):
        df, scores = data[split]
        for m in METHODS:
            leads = per_engine_lead_times(df, scores[m] > thr[m])
            print(f"{split} / {m}: {[int(u) for u in leads[leads.isna()].index]}")

    print()
    plot_roc_pr(data, ops)
    plot_zone_rates(data, thr)
    plot_lead_times(data, thr)
    plot_confusion(data, thr)
    print(f"saved: {METRICS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()