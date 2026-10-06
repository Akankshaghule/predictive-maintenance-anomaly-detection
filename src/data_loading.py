from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_cmapss(subset: str = "FD001"):
    """Load train, test and RUL files for one C-MAPSS subset."""
    train = pd.read_csv(RAW_DIR / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(RAW_DIR / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul_at_end"])
    rul.index = rul.index + 1          # engine IDs start at 1
    rul.index.name = "unit"
    return train, test, rul


def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """In training data every engine runs to failure, so RUL = last cycle - current cycle."""
    train = train.copy()
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = max_cycle - train["cycle"]
    return train


def summarize(train, test, rul):
    print("=== SHAPES ===")
    print(f"train: {train.shape} | test: {test.shape} | rul: {rul.shape}")

    print("\n=== FIRST 3 ROWS (train) ===")
    print(train.head(3).to_string())

    lifetimes = train.groupby("unit")["cycle"].max()
    print("\n=== ENGINE LIFETIMES IN TRAIN (cycles) ===")
    print(f"engines: {lifetimes.shape[0]} | min: {lifetimes.min()} | "
          f"max: {lifetimes.max()} | mean: {lifetimes.mean():.1f}")

    print("\n=== DATA QUALITY ===")
    print(f"missing values: {int(train.isna().sum().sum())}")
    print(f"duplicate rows: {int(train.duplicated().sum())}")

    constant = [c for c in SETTING_COLS + SENSOR_COLS if train[c].nunique() == 1]
    print(f"constant columns (carry no information): {constant}")


if __name__ == "__main__":
    train_df, test_df, rul_df = load_cmapss("FD001")
    train_df = add_train_rul(train_df)
    summarize(train_df, test_df, rul_df)