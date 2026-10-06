import numpy as np
import pandas as pd

from src.config import (SENSORS, HEALTHY_RUL, PROCESSED_DIR, METRICS_DIR,
                        FEATURE_WINDOW, RATE_LAG)

META_COLS = ["unit", "cycle", "rul", "is_anomaly"]

# group -> suffixes appended to each sensor name ("" means the raw scaled sensor)
FEATURE_GROUPS = {
    "raw": [""],
    "trend": ["_rmean", "_rstd", "_rate"],
    "stats": ["_rms", "_peak", "_kurt", "_skew"],
}


def get_feature_names(groups=("raw", "trend", "stats")):
    """Column names for the chosen feature groups."""
    return [f"{s}{suffix}" for g in groups for suffix in FEATURE_GROUPS[g] for s in SENSORS]


def feature_group_map():
    return {f"{s}{suffix}": g for g, suffixes in FEATURE_GROUPS.items()
            for suffix in suffixes for s in SENSORS}


def _rolling(frame, units, window, how):
    """Backward-looking rolling statistic computed separately for each engine."""
    return frame.groupby(units).transform(
        lambda s: getattr(s.rolling(window, min_periods=window), how)())


def build_features(df, window=FEATURE_WINDOW, lag=RATE_LAG):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    frame, units = df[SENSORS], df["unit"]

    parts = [
        df[META_COLS],
        frame,
        _rolling(frame, units, window, "mean").add_suffix("_rmean"),
        _rolling(frame, units, window, "std").add_suffix("_rstd"),
        ((frame - frame.groupby(units).shift(lag)) / lag).add_suffix("_rate"),
        np.sqrt(_rolling(frame ** 2, units, window, "mean")).add_suffix("_rms"),
        _rolling(frame.abs(), units, window, "max").add_suffix("_peak"),
        _rolling(frame, units, window, "kurt").add_suffix("_kurt"),
        _rolling(frame, units, window, "skew").add_suffix("_skew"),
    ]
    out = pd.concat(parts, axis=1)

    out = out[df["cycle"] >= window]                      # drop warm-up rows
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)  # flat windows give undefined kurtosis/skew
    return out.reset_index(drop=True)


def separation_table(train_feat):
    """How far does each feature move from healthy to anomaly, in healthy-std units?"""
    cols = get_feature_names()
    healthy = train_feat[train_feat["rul"] > HEALTHY_RUL][cols]
    anomaly = train_feat[train_feat["is_anomaly"] == 1][cols]
    sd = healthy.std().replace(0, 1)
    shift = ((anomaly.mean() - healthy.mean()) / sd).abs()
    table = shift.rename("abs_shift").to_frame()
    table["group"] = table.index.map(feature_group_map())
    return table.sort_values("abs_shift", ascending=False)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    feats = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        feats[name] = build_features(df)
        feats[name].to_csv(PROCESSED_DIR / f"features_{name}.csv", index=False)
        print(f"{name:<6} rows {len(df):>6} -> {len(feats[name]):>6} | "
              f"anomaly rows kept: {int(feats[name]['is_anomaly'].sum())}")

    n_features = len(get_feature_names())
    print(f"\nTotal features: {n_features} ({len(SENSORS)} sensors x 8 feature types)")

    table = separation_table(feats["train"])
    table.to_csv(METRICS_DIR / "feature_separation.csv")

    print("\n=== AVERAGE SEPARATION BY FEATURE TYPE (healthy-std units) ===")
    by_type = table.copy()
    by_type["type"] = [n.split("_", 1)[1] if "_" in n else "raw" for n in by_type.index]
    print(by_type.groupby("type")["abs_shift"].mean().sort_values(ascending=False).round(2).to_string())

    print("\n=== TOP 10 INDIVIDUAL FEATURES ===")
    print(table.head(10).round(2).to_string())
    print(f"\nsaved: {PROCESSED_DIR} (features_train.csv, features_val.csv, features_test.csv)")


if __name__ == "__main__":
    main()