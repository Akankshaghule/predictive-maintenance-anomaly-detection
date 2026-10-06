import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets the dashboard import from src/

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import FEATURE_SETS
from src.config import (HEALTH_BANDS, IF_N_ESTIMATORS, METRICS_DIR, MODELS_DIR,
                        PROCESSED_DIR, SENSORS)

st.set_page_config(page_title="Predictive Maintenance Monitor", page_icon="🛠️", layout="wide")

# SENSOR_INFO = {
#     "s2": "LPC outlet temperature", "s3": "HPC outlet temperature", "s4": "LPT outlet temperature",
#     "s7": "HPC outlet pressure", "s8": "Fan speed", "s9": "Core speed",
#     "s11": "HPC static pressure", "s12": "Fuel flow / static pressure ratio",
#     "s13": "Corrected fan speed", "s14": "Corrected core speed", "s15": "Bypass ratio",
#     "s17": "Bleed enthalpy", "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
# }

from src.config import (HEALTH_BANDS, IF_N_ESTIMATORS, METRICS_DIR, MODELS_DIR,
                        PROCESSED_DIR, SENSOR_INFO, SENSORS)

BAND_COLORS = {"Healthy": "#bfe5c9", "Good": "#dcefb3", "Warning": "#fbdcae", "Critical": "#f1b5b5"}
UPPERS = [100] + [b for b, _ in HEALTH_BANDS[:-1]]


# ---------------------------------------------------------------- data
@st.cache_data
def load_split(split):
    health = pd.read_csv(PROCESSED_DIR / f"health_{split}.csv")
    feats = pd.read_csv(PROCESSED_DIR / f"features_{split}.csv", usecols=["unit", "cycle"] + SENSORS)
    z = feats.rename(columns={s: f"{s}_z" for s in SENSORS})
    df = health.merge(z, on=["unit", "cycle"], how="left")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    df[SENSORS] = scaler.inverse_transform(df[[f"{s}_z" for s in SENSORS]].to_numpy())
    return df


@st.cache_data
def load_reports():
    with open(METRICS_DIR / "isolation_forest_metrics.json") as f:
        im = json.load(f)
    comp = pd.read_csv(METRICS_DIR / "model_comparison.csv")
    return im, comp

@st.cache_data
def load_alerts():
    path = ROOT / "results" / "alerts.csv"
    return pd.read_csv(path) if path.exists() else None

# ---------------------------------------------------------------- charts
def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def add_truth(fig, d, show_truth):
    if show_truth:
        t = d[d["is_anomaly"] == 1]
        if len(t):
            fig.add_vrect(x0=t["cycle"].min(), x1=t["cycle"].max(),
                          fillcolor="red", opacity=0.08, line_width=0)


def gauge(health):
    steps = [{"range": [lo, hi], "color": BAND_COLORS[label]}
             for (lo, label), hi in zip(HEALTH_BANDS, UPPERS)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(health), number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#3b5b92"}, "steps": steps},
        title={"text": "Robot health"}))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def sensor_chart(d, sensor, show_truth):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["cycle"], y=d[sensor], mode="lines", line=dict(width=1.5)))
    bad = d[d["flag"]]
    fig.add_trace(go.Scatter(x=bad["cycle"], y=bad[sensor], mode="markers",
                             marker=dict(color="red", size=6)))
    add_truth(fig, d, show_truth)
    fig.update_layout(title=f"{sensor}: {SENSOR_INFO[sensor]}", height=280, showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Cycle")
    return fig


def score_chart(d, thr, show_truth):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["anomaly_score"], mode="lines", name="Anomaly score",
                             line=dict(width=1, color="#999999")))
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["smoothed_score"], mode="lines", name="Smoothed score",
                             line=dict(width=2)))
    bad = d[d["flag"]]
    fig.add_trace(go.Scatter(x=bad["cycle"], y=bad["anomaly_score"], mode="markers",
                             name="Flagged", marker=dict(color="red", size=6)))
    fig.add_hline(y=thr, line_dash="dash", line_color="red", annotation_text="alarm threshold")
    add_truth(fig, d, show_truth)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Cycle", yaxis_title="Anomaly score")
    return fig


def health_chart(d, show_truth):
    fig = go.Figure()
    for (lo, label), hi in zip(HEALTH_BANDS, UPPERS):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND_COLORS[label], opacity=0.5,
                      line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["health"], mode="lines", name="Health",
                             line=dict(width=2, color="#3b5b92")))
    add_truth(fig, d, show_truth)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis_title="Cycle", yaxis_title="Health score", yaxis_range=[0, 100])
    return fig


# ---------------------------------------------------------------- app
im, comp = load_reports()
cfg = im["config"]
thr = cfg["threshold"]

st.sidebar.title("Controls")
split_label = st.sidebar.radio("Dataset", ["Test engines", "Validation engines"])
split = "test" if split_label.startswith("Test") else "val"
data = load_split(split)
data["flag"] = data["anomaly_score"] > thr

units = sorted(int(u) for u in data["unit"].unique())
demo_units = sorted(int(u) for u in data.loc[data["is_anomaly"] == 1, "unit"].unique())
default_unit = demo_units[0] if demo_units else units[0]
unit = st.sidebar.selectbox("Engine (stand-in for a robot component)", units,
                            index=units.index(default_unit), key=f"unit_{split}")
eng = data[data["unit"] == unit].sort_values("cycle")
cmin, cmax = int(eng["cycle"].min()), int(eng["cycle"].max())
current = st.sidebar.slider("Simulated time (cycle)", cmin, cmax, cmax, key=f"cycle_{split}_{unit}")
sensors = st.sidebar.multiselect("Sensors to display", SENSORS, default=["s9", "s11", "s4"],
                                 format_func=lambda s: f"{s}: {SENSOR_INFO[s]}")
show_truth = st.sidebar.checkbox("Show ground truth (demo only)", value=False)

d = eng[eng["cycle"] <= current]
now = d.iloc[-1]

st.title("🛠️ AI-Based Predictive Maintenance and Anomaly Detection")
st.caption("Data: NASA C-MAPSS FD001 turbofan simulation, used as a stand-in for robot component "
           "sensor data. This is not a real robot.")

# 1. Overall health
st.header("1. Overall health")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    show_chart(gauge(now["health"]))
prev = d["health"].iloc[-11] if len(d) > 10 else d["health"].iloc[0]
c2.metric("Health score", f"{now['health']:.1f} / 100", delta=f"{now['health'] - prev:+.1f} vs 10 cycles ago")
c2.metric("Status", str(now["status"]).upper())
c3.metric("Current reading", "ANOMALY" if now["flag"] else "Normal")
c3.metric("Cycle", f"{int(now['cycle'])} of {cmax}")
if show_truth:
    c3.caption(f"Ground truth: {int(now['rul'])} cycles before failure")

# 2. Sensor monitoring
st.header("2. Sensor monitoring")
if sensors:
    cols = st.columns(2)
    for i, s in enumerate(sensors):
        with cols[i % 2]:
            show_chart(sensor_chart(d, s, show_truth))
    st.caption("Red dots mark cycles the model flagged as anomalous.")
else:
    st.info("Pick at least one sensor in the sidebar.")

# 3. Anomaly detection
st.header("3. Anomaly detection")
show_chart(score_chart(d, thr, show_truth))
n_flag = int(d["flag"].sum())
st.write(f"Flagged so far: **{n_flag}** of {len(d)} cycles ({n_flag / len(d):.1%}). "
         f"Alarm threshold: **{thr:.4f}**")

# 4. Alerts
st.header("4. Alerts")
status = now["status"]
if status == "Critical":
    st.error("🔴 Critical: high anomaly score / possible component failure")
elif status == "Warning":
    st.warning("⚠️ Warning: abnormal sensor behavior detected")
else:
    st.success("✅ No alert: operating within the normal range")
if status in ("Warning", "Critical"):
    dev = sorted(SENSORS, key=lambda s: -abs(now[f"{s}_z"]))[:3]
    txt = ", ".join(f"{s} ({SENSOR_INFO[s]}) {now[f'{s}_z']:+.1f}σ" for s in dev)
    st.write(f"Largest deviations from healthy behaviour: {txt}")

st.subheader("Alert log")
alerts = load_alerts()
if alerts is None:
    st.info("No alert log found. Run: python -m src.alert_engine")
else:
    log = alerts[(alerts["split"] == split) & (alerts["unit"] == unit) & (alerts["cycle"] <= current)]
    if log.empty:
        st.write("No alerts raised yet for this engine up to the selected cycle.")
    else:
        log = log.sort_values("cycle", ascending=False).copy()
        log["severity"] = log["severity"].map({"Warning": "⚠️ Warning", "Critical": "🔴 Critical"})
        st.dataframe(log[["sim_time", "cycle", "severity", "sensor", "anomaly_score", "health", "message"]])
    st.caption("sim_time is a simulated clock (1 cycle = 1 hour); the dataset has no real timestamps. "
               "The log needs 3 consecutive cycles at a severity before alerting, so it can lag the live banner above.")

# 5. Historical trends
st.header("5. Health trend")
show_chart(health_chart(d, show_truth))

# 6. Model information
st.header("6. Model information")
left, right = st.columns(2)
with left:
    fs = cfg["feature_set"]
    st.markdown(
        f"- **Model:** Isolation Forest ({IF_N_ESTIMATORS} trees), trained on healthy data only\n"
        f"- **Feature set:** {fs} ({len(FEATURE_SETS[fs])} features)\n"
        f"- **Alarm threshold:** {thr:.4f} = {cfg['quantile'] * 100:g}th percentile of healthy validation scores\n"
        f"- **Observations in this dataset:** {len(data):,} rows from {len(units)} engines\n"
        f"- **Anomalies flagged (all engines):** {int(data['flag'].sum()):,}")
with right:
    tm, tl = im["test"]["metrics"], im["test"]["lead_time"]
    st.markdown("**Held-out test engines** (never used to fit or tune)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{tm['precision']:.3f}")
    m2.metric("Recall", f"{tm['recall']:.3f}")
    m3.metric("F1-score", f"{tm['f1']:.3f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    m5.metric("False alarms on healthy data", f"{tm['healthy_false_alarm_rate']:.1%}")
    m6.metric("Engines detected", f"{tl['detected']}/{tl['engines']}")
    st.caption(f"Median detection lead time: {tl['median_lead']:.0f} cycles before failure. "
               "Precision is low partly because early warnings (31 to 100 cycles out) count as false "
               "positives under the 30-cycle label.")
with st.expander("Comparison with the z-score baseline (same alarm calibration)"):
    st.dataframe(comp.round(3))