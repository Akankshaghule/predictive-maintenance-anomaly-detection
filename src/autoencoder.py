import json

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection import FEATURE_SETS
from src.config import (AE_BATCH_SIZE, AE_EPOCHS, AE_FEATURE_SETS, AE_HIDDEN, AE_LATENTS, AE_LR,
                        AE_PATIENCE, HEALTHY_RUL, METRICS_DIR, MODELS_DIR, PROCESSED_DIR,
                        RANDOM_SEED, ROOT)
from src.evaluation import evaluate, lead_time_summary, per_engine_lead_times
from src.model_evaluation import load_split

FIG_DIR = ROOT / "results" / "figures"
Q_TARGET = 0.995      # same calibration as Stage 7
METHODS = ["Z-score baseline", "Isolation Forest", "Autoencoder"]


class Autoencoder(nn.Module):
    def __init__(self, d, hidden, latent):
        super().__init__()
        enc, prev = [], d
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc.append(nn.Linear(prev, latent))
        dec, prev = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, d))
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(x_train, x_val, d, latent):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model = Autoencoder(d, AE_HIDDEN, latent)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(x_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)

    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(AE_EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), AE_BATCH_SIZE):
            batch = xt[perm[i:i + AE_BATCH_SIZE]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), xv).item()
        history.append((total / len(xt), val_loss))
        if val_loss < best - 1e-5:
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= AE_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, history


def recon_error(model, x):
    """Per-row mean squared reconstruction error = anomaly score (higher = more abnormal)."""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        return ((model(t) - t) ** 2).mean(dim=1).numpy()


def fit_candidate(train, val_df, healthy_val, fs_name, latent):
    feats = FEATURE_SETS[fs_name]
    healthy_train = train[train["rul"] > HEALTHY_RUL][feats].to_numpy()
    scaler = StandardScaler().fit(healthy_train)
    x_val_all = scaler.transform(val_df[feats].to_numpy())
    model, history = train_ae(scaler.transform(healthy_train), x_val_all[healthy_val], len(feats), latent)
    return {"fs": fs_name, "latent": latent, "feats": feats, "scaler": scaler, "model": model,
            "history": history, "val_scores": recon_error(model, x_val_all)}


def plot_training(history, label):
    h = np.array(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h[:, 0], label="train loss (healthy engines)")
    ax.plot(h[:, 1], label="validation loss (healthy rows)")
    ax.set(yscale="log", xlabel="Epoch", ylabel="MSE", title=f"Autoencoder training ({label})")
    ax.legend()
    path = FIG_DIR / "10_ae_training.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_comparison(table):
    metrics = ["precision", "recall", "f1", "auc"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        sub = table[table["split"] == split].set_index("method")[metrics].loc[METHODS]
        sub.T.plot(kind="bar", ax=ax, rot=0)
        ax.set(title=f"Model comparison ({split}, matched calibration)", ylim=(0, 1.05))
    fig.tight_layout()
    path = FIG_DIR / "11_model_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "baseline_metrics.json") as f:
        k = json.load(f)["tuned|val"]["k"]

    train = pd.read_csv(PROCESSED_DIR / "features_train.csv")
    data = {s: load_split(s, k) for s in ("val", "test")}      # (features df, {method: scores})
    val_df = data["val"][0]
    healthy_val = (val_df["rul"] > HEALTHY_RUL).to_numpy()

    # ---- candidate search, judged on validation only ----
    print("Searching autoencoder candidates (validation only)...")
    cands, rows = [], []
    for fs_name in AE_FEATURE_SETS:
        for latent in AE_LATENTS:
            c = fit_candidate(train, val_df, healthy_val, fs_name, latent)
            thr = np.quantile(c["val_scores"][healthy_val], Q_TARGET)
            m = evaluate(val_df, c["val_scores"] > thr, c["val_scores"])
            c["val_auc"] = m["roc_auc"]
            cands.append(c)
            rows.append({"feature_set": fs_name, "n_features": len(c["feats"]), "latent": latent,
                         "epochs": len(c["history"]), "val_auc": m["roc_auc"], "val_f1": m["f1"],
                         "val_recall": m["recall"], "val_healthy_far": m["healthy_false_alarm_rate"]})
    grid = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    grid.to_csv(METRICS_DIR / "ae_grid.csv", index=False)
    print("\n=== AUTOENCODER CANDIDATES (ranked by validation AUC) ===")
    print(grid.round(3).to_string(index=False))

    best = max(cands, key=lambda c: c["val_auc"])
    model, scaler, feats = best["model"], best["scaler"], best["feats"]
    label = f"{best['fs']}, bottleneck {best['latent']}"
    print(f"\nSelected on validation: {label}")
    plot_training(best["history"], label)

    for split, (df, scores) in data.items():
        scores["Autoencoder"] = recon_error(model, scaler.transform(df[feats].to_numpy()))
        (df[["unit", "cycle", "rul", "is_anomaly"]]
         .assign(ae_score=scores["Autoencoder"])
         .to_csv(PROCESSED_DIR / f"ae_scores_{split}.csv", index=False))

    # ---- same label-free calibration for every method ----
    val_scores = data["val"][1]
    thr = {m: float(np.quantile(val_scores[m][healthy_val], Q_TARGET)) for m in METHODS}
    rows = []
    for split in ("val", "test"):
        df, scores = data[split]
        for m in METHODS:
            flags = scores[m] > thr[m]
            met, lt = evaluate(df, flags, scores[m]), lead_time_summary(df, flags)
            rows.append({"split": split, "method": m, "precision": met["precision"],
                         "recall": met["recall"], "f1": met["f1"], "auc": met["roc_auc"],
                         "far": met["false_alarm_rate"], "healthy_far": met["healthy_false_alarm_rate"],
                         "detected": f"{lt['detected']}/{lt['engines']}", "median_lead": lt["median_lead"]})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS_DIR / "ae_comparison.csv", index=False)
    print(f"\n=== THREE-WAY COMPARISON ({Q_TARGET} percentile of healthy validation scores) ===")
    print(table.round(3).to_string(index=False))

    print("\n=== ENGINES MISSED ===")
    for split in ("val", "test"):
        df, scores = data[split]
        for m in METHODS:
            leads = per_engine_lead_times(df, scores[m] > thr[m])
            print(f"{split} / {m}: {[int(u) for u in leads[leads.isna()].index]}")

    torch.save({"state_dict": model.state_dict(), "input_dim": len(feats), "hidden": list(AE_HIDDEN),
                "latent": best["latent"], "features": feats, "feature_set": best["fs"],
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "threshold": thr["Autoencoder"], "threshold_quantile": Q_TARGET},
               MODELS_DIR / "autoencoder.pth")
    print()
    plot_comparison(table)
    print(f"saved: {MODELS_DIR / 'autoencoder.pth'}")
    print(f"saved: {METRICS_DIR / 'ae_comparison.csv'} and ae_grid.csv")


if __name__ == "__main__":
    main()