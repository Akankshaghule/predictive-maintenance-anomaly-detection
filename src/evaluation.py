import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.config import HEALTHY_RUL, ALERT_PERSIST


def evaluate(df, flags, scores=None):
    """Row-level metrics. df needs is_anomaly and rul columns; flags are 0/1 predictions."""
    y = df["is_anomaly"].to_numpy()
    pred = np.asarray(flags).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    healthy = (df["rul"] > HEALTHY_RUL).to_numpy()
    auc = np.nan
    if scores is not None and len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, scores))
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "healthy_false_alarm_rate": float(pred[healthy].mean()) if healthy.any() else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def lead_time_summary(df, flags, persist=ALERT_PERSIST):
    """Cycles before failure at which each engine first raised a persistent alarm.

    Only engines whose data reaches the anomaly zone are included.
    Alarms during the healthy phase (RUL > HEALTHY_RUL) are ignored here;
    they show up in healthy_false_alarm_rate instead.
    """
    d = df[["unit", "cycle", "rul", "is_anomaly"]].copy()
    d["flag"] = np.asarray(flags).astype(int)
    leads = {}
    for unit, g in d.groupby("unit"):
        if not g["is_anomaly"].any():
            continue
        g = g.sort_values("cycle")
        persistent = g["flag"].rolling(persist).min().fillna(0).astype(bool)
        hit = g[persistent & (g["rul"] <= HEALTHY_RUL)]
        leads[unit] = hit["rul"].max() if len(hit) else np.nan   # earliest alarm = largest RUL
    s = pd.Series(leads, dtype=float)
    found = s.dropna()
    return {
        "engines": int(len(s)),
        "detected": int(len(found)),
        "median_lead": float(found.median()) if len(found) else np.nan,
        "mean_lead": float(found.mean()) if len(found) else np.nan,
        "min_lead": float(found.min()) if len(found) else np.nan,
    }

def per_engine_lead_times(df, flags, persist=ALERT_PERSIST):
    """Same logic as lead_time_summary, but returns one value per engine (NaN = missed)."""
    d = df[["unit", "cycle", "rul", "is_anomaly"]].copy()
    d["flag"] = np.asarray(flags).astype(int)
    leads = {}
    for unit, g in d.groupby("unit"):
        if not g["is_anomaly"].any():
            continue
        g = g.sort_values("cycle")
        persistent = g["flag"].rolling(persist).min().fillna(0).astype(bool)
        hit = g[persistent & (g["rul"] <= HEALTHY_RUL)]
        leads[unit] = hit["rul"].max() if len(hit) else np.nan
    return pd.Series(leads, dtype=float)