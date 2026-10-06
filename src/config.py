from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"

SUBSET = "FD001"

# 14 informative sensors (constant sensors and s6 removed, see EDA)
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
           "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

HEALTHY_RUL = 100    # rows with more cycles left than this = healthy (used for fitting)
ANOMALY_RUL = 30     # rows with this many cycles left or fewer = anomaly (evaluation label)
VAL_FRACTION = 0.2   # share of training engines held out for validation
RANDOM_SEED = 42

FEATURE_WINDOW = 10   # number of past cycles used for rolling features
RATE_LAG = 5          # rate of change = (value now - value 5 cycles ago) / 5

ALERT_PERSIST = 3     # consecutive flagged cycles needed to raise a persistent alarm

IF_N_ESTIMATORS = 300                              # number of trees in the forest
THRESHOLD_QUANTILES = [0.95, 0.97, 0.99, 0.995]    # healthy-score percentile used as alarm threshold

HEALTH_SMOOTH = 5            # cycles of causal averaging applied to anomaly scores
CRITICAL_ZONE_RUL = 15       # "imminent failure" zone used to anchor the low end of the scale

# (population, percentile of its smoothed scores, health value) -> piecewise-linear scale
HEALTH_SCALE = [
    ("healthy", 50, 100),
    ("healthy", 99.5, 70),
    ("critical", 50, 40),
    ("critical", 95, 0),
]

# (minimum health, status), highest band first
HEALTH_BANDS = [(90, "Healthy"), (70, "Good"), (40, "Warning"), (0, "Critical")]

SIM_START = "2026-01-01 00:00"   # simulated clock for display only; the dataset has no real timestamps
SIM_HOURS_PER_CYCLE = 1          # assumption: one cycle = one hour

SENSOR_INFO = {
    "s2": "LPC outlet temperature", "s3": "HPC outlet temperature", "s4": "LPT outlet temperature",
    "s7": "HPC outlet pressure", "s8": "Fan speed", "s9": "Core speed",
    "s11": "HPC static pressure", "s12": "Fuel flow / static pressure ratio",
    "s13": "Corrected fan speed", "s14": "Corrected core speed", "s15": "Bypass ratio",
    "s17": "Bleed enthalpy", "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
}

AE_HIDDEN = (32, 16)      # encoder hidden layer sizes (the decoder mirrors them)
AE_FEATURE_SETS = ["raw", "rmean", "rms+peak"]   # candidate input feature sets
AE_LATENTS = [3, 6]                              # candidate bottleneck sizes        
AE_EPOCHS = 150           # maximum epochs
AE_BATCH_SIZE = 128
AE_LR = 1e-3
AE_PATIENCE = 15          # stop if healthy-validation loss has not improved for this many epochs