import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (CRITICAL_ZONE_RUL, HEALTH_BANDS, HEALTH_SCALE, HEALTH_SMOOTH,
                        HEALTHY_RUL, MODELS_DIR, PROCESSED_DIR, ROOT)

FIG_DIR = ROOT / "results" / "figures"
ZONE_BINS = [-1, 15, 30, 50, 75, 100, 1e9]
ZONE_LABELS = ["0-15", "16-30", "31-50", "51-75", "76-100", ">100"]
ZONE_ORDER = ZONE_LABELS[::-1]
STATUS_ORDER = [label for _, label in HEALTH_BANDS][::-1]      # Healthy ... Critical -> reversed below
STATUS_COLORS = {"Healthy": "#2e9e4f", "Good": "#9acd32", "Warning": "#f0a030", "Critical": "#d64545"}


def add_smoothed(df, window=HEALTH_SMOOTH):
    """Causal moving average per engine (only current and past cycles)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    df["smoothed_score"] = df.groupby("unit")["anomaly_score"].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    return df


def fit_health_scale(ref):
    """Turn the anchors in HEALTH_SCALE into actual score values using reference data."""
    pops = {"healthy": ref.loc[ref["rul"] > HEALTHY_RUL, "smoothed_score"],
            "critical": ref.loc[ref["rul"] <= CRITICAL_ZONE_RUL, "smoothed_score"]}
    xs = [float(np.percentile(pops[p], pct)) for p, pct, _ in HEALTH_SCALE]
    ys = [float(h) for _, _, h in HEALTH_SCALE]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"Anchor scores must increase, got {xs}. Check HEALTH_SCALE in config.py.")
    return {"scores": xs, "health": ys}


def health_from_score(score, scale):
    return np.round(np.interp(score, scale["scores"], scale["health"]), 1)


def score_at_health(h, scale):
    """Inverse mapping: which smoothed score corresponds to a given health value."""
    return float(np.interp(h, scale["health"][::-1], scale["scores"][::-1]))


def status_from_health(health):
    health = np.asarray(health)
    out = np.full(len(health), HEALTH_BANDS[-1][1], dtype=object)
    for min_h, label in reversed(HEALTH_BANDS):      # lowest band first, higher bands override
        out[health >= min_h] = label
    return out


def add_health(df, scale):
    df = add_smoothed(df)
    df["health"] = health_from_score(df["smoothed_score"].to_numpy(), scale)
    df["status"] = status_from_health(df["health"].to_numpy())
    return df


def zone_tables(df):
    zone = pd.cut(df["rul"], bins=ZONE_BINS, labels=ZONE_LABELS)
    mean_health = df.groupby(zone, observed=False)["health"].mean().reindex(ZONE_ORDER)
    shares = (pd.crosstab(zone, df["status"], normalize="index")
              .reindex(index=ZONE_ORDER, columns=[l for _, l in HEALTH_BANDS]).fillna(0))
    return mean_health, shares


def rank_correlation(df):
    """Per engine (reaching the anomaly zone): Spearman correlation between health and RUL."""
    rhos = []
    for _, g in df.groupby("unit"):
        if g["is_anomaly"].any() and g["health"].nunique() > 1:
            rhos.append(g["health"].corr(g["rul"], method="spearman"))
    return pd.Series(rhos).dropna()


def plot_trajectories(val, test):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    uppers = [100] + [b for b, _ in HEALTH_BANDS[:-1]]
    for ax, (name, df) in zip(axes, (("validation", val), ("test", test))):
        for (lo, label), hi in zip(HEALTH_BANDS, uppers):
            ax.axhspan(lo, hi, color=STATUS_COLORS[label], alpha=0.12)
        units = [u for u, g in df.groupby("unit") if g["is_anomaly"].any()][:6]
        for u in units:
            g = df[df["unit"] == u]
            ax.plot(-g["rul"], g["health"], lw=1.3, label=f"engine {u}")
        ax.set(title=f"Health score over time ({name} engines)", xlabel="Cycles before failure (0 = failure)",
               ylabel="Health score", ylim=(0, 100))
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "09_health_trajectories.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    val = add_smoothed(pd.read_csv(PROCESSED_DIR / "scores_val.csv"))
    test = pd.read_csv(PROCESSED_DIR / "scores_test.csv")

    scale = fit_health_scale(val)
    val, test = add_health(val, scale), add_health(test, scale)

    print("=== HEALTH SCALE (fitted on validation engines) ===")
    for (pop, pct, h), x in zip(HEALTH_SCALE, scale["scores"]):
        print(f"  health {h:>5.0f}  <-  smoothed score {x:.4f}   ({pop} p{pct:g})")
    print("  score at band boundaries: " +
          ", ".join(f"health {b} -> {score_at_health(b, scale):.4f}" for b, _ in HEALTH_BANDS[:-1]))

    for name, df in (("validation", val), ("test", test)):
        mean_health, shares = zone_tables(df)
        print(f"\n=== {name.upper()}: mean health and status share by cycles-before-failure zone ===")
        out = shares.round(3)
        out.insert(0, "mean_health", mean_health.round(1))
        print(out.to_string())
        rho = rank_correlation(df)
        print(f"per-engine Spearman(health, RUL): mean {rho.mean():.3f}, min {rho.min():.3f} "
              f"({len(rho)} engines)")

    cols = ["unit", "cycle", "rul", "is_anomaly", "anomaly_score", "smoothed_score", "health", "status"]
    val[cols].to_csv(PROCESSED_DIR / "health_val.csv", index=False)
    test[cols].to_csv(PROCESSED_DIR / "health_test.csv", index=False)
    with open(MODELS_DIR / "health_scale.json", "w") as f:
        json.dump({**scale, "smooth_window": HEALTH_SMOOTH, "bands": HEALTH_BANDS}, f, indent=2)
    print(f"\nsaved: {PROCESSED_DIR / 'health_val.csv'} and health_test.csv")
    print(f"saved: {MODELS_DIR / 'health_scale.json'}")
    plot_trajectories(val, test)


if __name__ == "__main__":
    main()