import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (ALERT_PERSIST, HEALTHY_RUL, METRICS_DIR, PROCESSED_DIR, ROOT, SENSORS,
                        SENSOR_INFO, SIM_HOURS_PER_CYCLE, SIM_START)

ALERTS_PATH = ROOT / "results" / "alerts.csv"
LEVELS = {"Warning": 1, "Critical": 2}
LEVEL_NAMES = {1: "Warning", 2: "Critical"}
SIM_START_DT = datetime.strptime(SIM_START, "%Y-%m-%d %H:%M")
EVENT_COLS = ["split", "unit", "cycle", "sim_time", "severity", "sensor", "sensor_deviation_sigma",
              "top_sensors", "anomaly_score", "health", "message"]


def load_health(split):
    h = pd.read_csv(PROCESSED_DIR / f"health_{split}.csv")
    f = pd.read_csv(PROCESSED_DIR / f"features_{split}.csv", usecols=["unit", "cycle"] + SENSORS)
    f = f.rename(columns={s: f"{s}_z" for s in SENSORS})
    return h.merge(f, on=["unit", "cycle"], how="left").sort_values(["unit", "cycle"]).reset_index(drop=True)


def add_levels(df, persist=ALERT_PERSIST):
    """level: severity of this cycle. sustained: lowest level over the last N cycles
    (so a level counts only if it held for N cycles). recent_max: highest level over the last N."""
    df = df.copy()
    df["level"] = df["status"].map(LEVELS).fillna(0).astype(int)
    g = df.groupby("unit")["level"]
    df["sustained"] = g.transform(lambda s: s.rolling(persist).min()).fillna(0).astype(int)
    df["recent_max"] = g.transform(lambda s: s.rolling(persist, min_periods=1).max()).astype(int)
    return df


def make_event(row, split, level):
    z = {s: float(getattr(row, f"{s}_z")) for s in SENSORS}
    top = sorted(SENSORS, key=lambda s: -abs(z[s]))[:3]
    sensor = top[0]
    severity = LEVEL_NAMES[level]
    head = ("Abnormal sensor behavior detected" if severity == "Warning"
            else "High anomaly score, possible component failure")
    message = (f"{head}. Largest deviation: {sensor} ({SENSOR_INFO[sensor]}) "
               f"{z[sensor]:+.1f} sigma from healthy behavior.")
    sim_time = SIM_START_DT + timedelta(hours=SIM_HOURS_PER_CYCLE * (int(row.cycle) - 1))
    return {
        "split": split, "unit": int(row.unit), "cycle": int(row.cycle),
        "sim_time": sim_time.strftime("%Y-%m-%d %H:%M"), "severity": severity,
        "sensor": sensor, "sensor_deviation_sigma": round(z[sensor], 2),
        "top_sensors": "; ".join(f"{s} {z[s]:+.1f}" for s in top),
        "anomaly_score": round(float(row.smoothed_score), 4), "health": float(row.health),
        "message": message,
    }


def build_alerts(df, split):
    """One event each time an engine's sustained severity rises above its current alert state."""
    events = []
    for _, g in df.groupby("unit"):
        state = 0
        for row in g.itertuples(index=False):
            if row.sustained > state:
                state = int(row.sustained)
                events.append(make_event(row, split, state))
            elif row.recent_max < state:       # condition has cleared; allow future alerts
                state = int(row.recent_max)
    return pd.DataFrame(events, columns=EVENT_COLS)


def evaluate_alerts(df, events):
    """How early alerts fire (ground truth RUL used only here) and how many fire while healthy."""
    lead_w, lead_c = {}, {}
    for unit, g in df.groupby("unit"):
        if g["is_anomaly"].any():
            d = g[g["rul"] <= HEALTHY_RUL]
            w, c = d[d["sustained"] >= 1], d[d["sustained"] >= 2]
            lead_w[unit] = w["rul"].max() if len(w) else np.nan
            lead_c[unit] = c["rul"].max() if len(c) else np.nan
    lw, lc = pd.Series(lead_w, dtype=float), pd.Series(lead_c, dtype=float)

    ev = events.merge(df[["unit", "cycle", "rul"]], on=["unit", "cycle"], how="left")
    early = ev[ev["rul"] > HEALTHY_RUL]
    med = lambda s: float(s.dropna().median()) if s.notna().any() else np.nan
    return {
        "engines": int(df["unit"].nunique()),
        "engines_reaching_anomaly_zone": int(len(lw)),
        "warning_detected": int(lw.notna().sum()), "warning_median_lead": med(lw),
        "critical_detected": int(lc.notna().sum()), "critical_median_lead": med(lc),
        "alerts_total": int(len(events)),
        "alerts_in_healthy_phase": int(len(early)),
        "engines_with_healthy_phase_alert": int(early["unit"].nunique()),
    }


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_events, report = [], {}
    for split in ("val", "test"):
        df = add_levels(load_health(split))
        events = build_alerts(df, split)
        all_events.append(events)
        report[split] = evaluate_alerts(df, events)

    alerts = pd.concat(all_events, ignore_index=True)
    alerts.to_csv(ALERTS_PATH, index=False)
    with open(METRICS_DIR / "alert_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== ALERT LOG ===")
    for split in ("val", "test"):
        a = alerts[alerts["split"] == split]
        print(f"{split}: {len(a)} alerts "
              f"({int((a['severity'] == 'Warning').sum())} Warning, {int((a['severity'] == 'Critical').sum())} Critical)")

    print("\n=== ALERT ENGINE PERFORMANCE ===")
    for split in ("val", "test"):
        r = report[split]
        n = r["engines_reaching_anomaly_zone"]
        print(f"{split}: Warning alert before failure in {r['warning_detected']}/{n} engines "
              f"(median lead {r['warning_median_lead']:.0f} cycles) | "
              f"Critical in {r['critical_detected']}/{n} (median lead {r['critical_median_lead']:.0f} cycles)")
        print(f"      alerts raised while still healthy (>{HEALTHY_RUL} cycles from failure): "
              f"{r['alerts_in_healthy_phase']} across {r['engines_with_healthy_phase_alert']} of {r['engines']} engines")

    print("\n=== SAMPLE ALERTS ===")
    cols = ["split", "unit", "cycle", "sim_time", "severity", "sensor", "anomaly_score", "health"]
    print(alerts[cols].head(8).to_string(index=False))
    print(f"\nsaved: {ALERTS_PATH}")
    print(f"saved: {METRICS_DIR / 'alert_evaluation.json'}")


if __name__ == "__main__":
    main()