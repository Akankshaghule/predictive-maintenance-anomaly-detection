import json

import numpy as np
import pandas as pd

from src.config import SENSORS, PROCESSED_DIR, METRICS_DIR
from src.evaluation import evaluate, lead_time_summary

Z_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
K_GRID = [1, 2, 3, 5, 7]
DEFAULT_CONFIG = (1, 3.0)   # naive rule: any single sensor beyond 3 standard deviations


def baseline_scores(df, k):
    """k-th largest |z| across sensors. 'score > z' means at least k sensors exceed z."""
    z = np.abs(df[SENSORS].to_numpy())
    z_desc = -np.sort(-z, axis=1)
    return z_desc[:, k - 1]


def run_config(df, k, z):
    scores = baseline_scores(df, k)
    flags = scores > z
    return evaluate(df, flags, scores), lead_time_summary(df, flags)


def run_grid(val):
    rows = []
    for k in K_GRID:
        for z in Z_GRID:
            m, _ = run_config(val, k, z)
            rows.append({"k": k, "z": z, **m})
    return pd.DataFrame(rows)


def show(name, m, lt):
    print(f"{name:<24} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
          f"AUC={m['roc_auc']:.3f} FAR={m['false_alarm_rate']:.3f} "
          f"healthyFAR={m['healthy_false_alarm_rate']:.3f} | "
          f"detected {lt['detected']}/{lt['engines']} engines, median lead {lt['median_lead']:.0f} cycles")


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(PROCESSED_DIR / "features_val.csv")
    test = pd.read_csv(PROCESSED_DIR / "features_test.csv")

    grid = run_grid(val).sort_values("f1", ascending=False).reset_index(drop=True)
    grid.to_csv(METRICS_DIR / "baseline_grid.csv", index=False)

    print("=== TOP 8 SETTINGS ON VALIDATION (by F1) ===")
    cols = ["k", "z", "precision", "recall", "f1", "false_alarm_rate", "healthy_false_alarm_rate"]
    print(grid[cols].head(8).round(3).to_string(index=False))

    best_k, best_z = int(grid.loc[0, "k"]), float(grid.loc[0, "z"])
    print(f"\nChosen on validation: k={best_k} sensors beyond z={best_z}")

    print("\n=== RESULTS ===")
    results = {}
    for label, (k, z) in {"naive (k=1, z=3)": DEFAULT_CONFIG, "tuned": (best_k, best_z)}.items():
        for split, df in (("val", val), ("test", test)):
            m, lt = run_config(df, k, z)
            show(f"{label} / {split}", m, lt)
            results[f"{label}|{split}"] = {"k": k, "z": z, "metrics": m, "lead_time": lt}

    with open(METRICS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {METRICS_DIR / 'baseline_grid.csv'}")
    print(f"saved: {METRICS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()