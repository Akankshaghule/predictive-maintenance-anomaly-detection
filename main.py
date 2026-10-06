"""Runs every stage of the project in order.

Requires train_FD001.txt, test_FD001.txt and RUL_FD001.txt in data/raw/.
Run from the project root:  python main.py
"""
import subprocess
import sys

STEPS = [
    "src.data_loading",
    "src.eda",
    "src.data_preprocessing",
    "src.feature_engineering",
    "src.baseline",
    "src.anomaly_detection",
    "src.model_evaluation",
    "src.health_score",
    "src.alert_engine",
    "src.autoencoder",
]


def main():
    for module in STEPS:
        print(f"\n===== python -m {module} =====", flush=True)
        result = subprocess.run([sys.executable, "-m", module])
        if result.returncode != 0:
            sys.exit(f"Step {module} failed. Fix it and run main.py again.")
    print("\nAll stages finished. Launch the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()