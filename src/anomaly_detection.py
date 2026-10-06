import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.baseline import show
from src.config import (HEALTHY_RUL, IF_N_ESTIMATORS, METRICS_DIR, MODELS_DIR,
                        PROCESSED_DIR, RANDOM_SEED, SENSORS, THRESHOLD_QUANTILES)
from src.evaluation import evaluate, lead_time_summary


def names(suffixes):
    return [f"{s}{suf}" for suf in suffixes for s in SENSORS]


FEATURE_SETS = {
    "raw": names([""]),
    "rmean": names(["_rmean"]),
    "rms+peak": names(["_rms", "_peak"]),
    "rmean+rms+peak": names(["_rmean", "_rms", "_peak"]),
    "raw+trend": names(["", "_rmean", "_rstd", "_rate"]),
    "all": names(["", "_rmean", "_rstd", "_rate", "_rms", "_peak", "_kurt", "_skew"]),
}


def train_iforest(train_df, features):
    """Fit on healthy rows only; failure labels are never used."""
    healthy = train_df[train_df["rul"] > HEALTHY_RUL]
    model = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination="auto",
                            random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(healthy[features].to_numpy())
    return model


def anomaly_score(model, df, features):
    """Higher = more abnormal (sklearn's score_samples is the opposite sign)."""
    return -model.score_samples(df[features].to_numpy())


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PROCESSED_DIR / "features_train.csv")
    val = pd.read_csv(PROCESSED_DIR / "features_val.csv")
    test = pd.read_csv(PROCESSED_DIR / "features_test.csv")
    healthy_val = (val["rul"] > HEALTHY_RUL).to_numpy()

    n_healthy = int((train["rul"] > HEALTHY_RUL).sum())
    print(f"Training on {n_healthy} healthy rows from {train['unit'].nunique()} engines\n")

    rows, fitted = [], {}
    for name, feats in FEATURE_SETS.items():
        model = train_iforest(train, feats)
        s_val = anomaly_score(model, val, feats)
        best = None
        for q in THRESHOLD_QUANTILES:
            thr = float(np.quantile(s_val[healthy_val], q))
            m = evaluate(val, s_val > thr, s_val)
            if best is None or m["f1"] > best["f1"]:
                best = {"q": q, "thr": thr, **m}
        fitted[name] = (model, feats, best)
        rows.append({
            "feature_set": name, "n_features": len(feats), "val_auc": best["roc_auc"],
            "quantile": best["q"], "val_precision": best["precision"],
            "val_recall": best["recall"], "val_f1": best["f1"],
            "val_healthy_far": best["healthy_false_alarm_rate"],
        })

    table = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    table.to_csv(METRICS_DIR / "isolation_forest_comparison.csv", index=False)
    print("=== FEATURE SETS (validation only; best percentile per set) ===")
    print(table.round(3).to_string(index=False))

    best_name = table.loc[0, "feature_set"]
    model, feats, best = fitted[best_name]
    thr = best["thr"]
    print(f"\nSelected on validation: feature set '{best_name}', "
          f"threshold = {best['q']:.3f} percentile of healthy scores ({thr:.4f})")

    print("\n=== FINAL RESULTS ===")
    final = {}
    for split, df in (("val", val), ("test", test)):
        s = anomaly_score(model, df, feats)
        flags = s > thr
        m, lt = evaluate(df, flags, s), lead_time_summary(df, flags)
        show(f"IsolationForest / {split}", m, lt)
        final[split] = {"metrics": m, "lead_time": lt}
        (df[["unit", "cycle", "rul", "is_anomaly"]]
         .assign(anomaly_score=s, flag=flags.astype(int))
         .to_csv(PROCESSED_DIR / f"scores_{split}.csv", index=False))

    with open(METRICS_DIR / "baseline_metrics.json") as f:
        baseline = json.load(f)
    for split in ("val", "test"):
        b = baseline[f"tuned|{split}"]
        show(f"Baseline tuned / {split}", b["metrics"], b["lead_time"])

    final["config"] = {"feature_set": best_name, "quantile": best["q"], "threshold": thr}
    with open(METRICS_DIR / "isolation_forest_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    joblib.dump({"model": model, "features": feats, "feature_set": best_name,
                 "threshold": thr, "threshold_quantile": best["q"]},
                MODELS_DIR / "isolation_forest.pkl")
    print(f"\nsaved: {MODELS_DIR / 'isolation_forest.pkl'}")
    print(f"saved: {PROCESSED_DIR / 'scores_val.csv'} and scores_test.csv")


if __name__ == "__main__":
    main()