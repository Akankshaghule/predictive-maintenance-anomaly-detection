import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (SUBSET, SENSORS, HEALTHY_RUL, ANOMALY_RUL, VAL_FRACTION,
                        RANDOM_SEED, PROCESSED_DIR, MODELS_DIR, METRICS_DIR)
from src.data_loading import load_cmapss, add_train_rul

KEEP_COLS = ["unit", "cycle", "rul", "is_anomaly"] + SENSORS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix types, fill gaps within each engine. Outliers are kept."""
    df = df.drop_duplicates().sort_values(["unit", "cycle"]).reset_index(drop=True)
    df[SENSORS] = df[SENSORS].apply(pd.to_numeric, errors="coerce")
    if df[SENSORS].isna().any().any():
        df[SENSORS] = df.groupby("unit")[SENSORS].transform(
            lambda s: s.interpolate(limit_direction="both"))
    return df


def add_test_rul(test: pd.DataFrame, rul: pd.DataFrame) -> pd.DataFrame:
    """Test engines stop early. RUL at the last cycle is given, so earlier rows are known too."""
    test = test.copy()
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    rul_at_end = test["unit"].map(rul["rul_at_end"])
    test["rul"] = rul_at_end + (last_cycle - test["cycle"])
    return test


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly"] = (df["rul"] <= ANOMALY_RUL).astype(int)
    return df


def split_by_engine(train: pd.DataFrame):
    """Hold out whole engines (never individual rows) for validation."""
    units = np.sort(train["unit"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    n_val = int(round(len(units) * VAL_FRACTION))
    val_units = np.sort(rng.choice(units, size=n_val, replace=False))
    is_val = train["unit"].isin(val_units)
    return train[~is_val].copy(), train[is_val].copy(), val_units


def fit_scaler(train_part: pd.DataFrame) -> StandardScaler:
    """Fit on healthy rows of training engines only, to avoid leakage."""
    healthy = train_part[train_part["rul"] > HEALTHY_RUL]
    scaler = StandardScaler()
    scaler.fit(healthy[SENSORS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[SENSORS] = scaler.transform(df[SENSORS])
    return out


def describe(name: str, df: pd.DataFrame):
    print(f"{name:<6} engines={df['unit'].nunique():>3}  rows={len(df):>6}  "
          f"anomaly_rows={int(df['is_anomaly'].sum()):>5} ({df['is_anomaly'].mean():.1%})")


def main():
    for d in (PROCESSED_DIR, MODELS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train, test, rul = load_cmapss(SUBSET)
    train = add_labels(add_train_rul(clean(train)))
    test = add_labels(add_test_rul(clean(test), rul))

    train_part, val_part, val_units = split_by_engine(train)

    scaler = fit_scaler(train_part)
    train_s = apply_scaler(train_part, scaler)
    val_s = apply_scaler(val_part, scaler)
    test_s = apply_scaler(test, scaler)

    train_s[KEEP_COLS].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_s[KEEP_COLS].to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_s[KEEP_COLS].to_csv(PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(METRICS_DIR / "split.json", "w") as f:
        json.dump({"val_units": [int(u) for u in val_units]}, f)

    print("=== SPLITS ===")
    describe("train", train_s)
    describe("val", val_s)
    describe("test", test_s)

    healthy_train = train_s[train_s["rul"] > HEALTHY_RUL][SENSORS]
    print("\n=== SCALING CHECK (healthy training rows) ===")
    print(f"mean of means: {healthy_train.mean().mean():.3f} | mean of stds: {healthy_train.std().mean():.3f}")

    near_fail = val_s[val_s["is_anomaly"] == 1][SENSORS]
    print("\n=== OUTLIERS KEPT (validation anomaly rows) ===")
    print(f"max |z-score| across sensors: {near_fail.abs().max().max():.1f}")

    print(f"\nsaved: {PROCESSED_DIR} (train.csv, val.csv, test.csv)")
    print(f"saved: {MODELS_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()