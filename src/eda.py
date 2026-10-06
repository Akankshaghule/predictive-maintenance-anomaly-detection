from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loading import load_cmapss, add_train_rul, SENSOR_COLS

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

HEALTHY_RUL = 100   # more than this many cycles left -> healthy
FAILING_RUL = 30    # this many cycles left or fewer -> near failure
PHASE_ORDER = ["healthy", "degrading", "near_failure"]


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def active_sensors(train):
    """Sensors that actually change (constant ones carry no information)."""
    return [c for c in SENSOR_COLS if train[c].nunique() > 1]


def label_phase(train):
    train = train.copy()
    train["phase"] = np.select(
        [train["rul"] > HEALTHY_RUL, train["rul"] <= FAILING_RUL],
        ["healthy", "near_failure"],
        default="degrading",
    )
    return train


def plot_trends(train, sensors, units=(1, 2, 3, 4, 5)):
    """Sensor value vs. cycles before failure for a few engines."""
    ncols = 3
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=True)
    axes = axes.ravel()
    for ax, s in zip(axes, sensors):
        for u in units:
            d = train[train["unit"] == u]
            ax.plot(-d["rul"], d[s], alpha=0.7, lw=1)
        ax.set_title(s)
    for ax in axes[len(sensors):]:
        ax.axis("off")
    fig.supxlabel("Cycles before failure (0 = failure)")
    fig.suptitle("Sensor trends for 5 engines", y=1.0)
    fig.tight_layout()
    save_fig(fig, "01_sensor_trends.png")


def plot_distributions(train, sensors):
    """Healthy vs near-failure histograms for every sensor."""
    healthy = train[train["phase"] == "healthy"]
    failing = train[train["phase"] == "near_failure"]
    ncols = 3
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = axes.ravel()
    for ax, s in zip(axes, sensors):
        ax.hist(healthy[s], bins=40, alpha=0.55, density=True, label="healthy")
        ax.hist(failing[s], bins=40, alpha=0.55, density=True, label="near failure")
        ax.set_title(s)
    for ax in axes[len(sensors):]:
        ax.axis("off")
    axes[0].legend()
    fig.suptitle("Healthy vs near-failure distributions", y=1.0)
    fig.tight_layout()
    save_fig(fig, "02_distributions.png")


def plot_correlation(train, sensors):
    corr = train[sensors + ["rul"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix (sensors and RUL)")
    save_fig(fig, "03_correlation.png")


def plot_boxplots(train, sensors):
    """Box plots of sensors standardised against healthy behaviour."""
    healthy = train[train["phase"] == "healthy"]
    mu = healthy[sensors].mean()
    sd = healthy[sensors].std().replace(0, 1)
    z = (train[sensors] - mu) / sd
    z["phase"] = train["phase"]
    long = z.melt(id_vars="phase", var_name="sensor", value_name="z_score")
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(data=long, x="sensor", y="z_score", hue="phase",
                hue_order=PHASE_ORDER, showfliers=False, ax=ax)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Sensor values in healthy-std units, by life phase")
    save_fig(fig, "04_boxplots.png")


def sensor_separation(train, sensors):
    """How far does each sensor move from healthy to near failure (in healthy std)?"""
    healthy = train[train["phase"] == "healthy"]
    failing = train[train["phase"] == "near_failure"]
    mu = healthy[sensors].mean()
    sd = healthy[sensors].std().replace(0, 1)
    shift = ((failing[sensors].mean() - mu) / sd).rename("shift_in_std")
    table = shift.to_frame()
    table["abs_shift"] = table["shift_in_std"].abs()
    table = table.sort_values("abs_shift", ascending=False)
    table.to_csv(MET_DIR / "sensor_separation.csv")
    return table


def main():
    train, _, _ = load_cmapss("FD001")
    train = label_phase(add_train_rul(train))
    sensors = active_sensors(train)

    print(f"Active sensors ({len(sensors)}): {sensors}")
    print("\nRows per phase:")
    print(train["phase"].value_counts().reindex(PHASE_ORDER).to_string())

    print("\nSummary statistics (active sensors):")
    print(train[sensors].describe().T[["mean", "std", "min", "max"]].round(3).to_string())

    plot_trends(train, sensors)
    plot_distributions(train, sensors)
    plot_correlation(train, sensors)
    plot_boxplots(train, sensors)

    table = sensor_separation(train, sensors)
    print("\nSensor shift from healthy to near failure (in healthy std units):")
    print(table.round(2).to_string())
    print(f"\nsaved: {MET_DIR / 'sensor_separation.csv'}")


if __name__ == "__main__":
    main()