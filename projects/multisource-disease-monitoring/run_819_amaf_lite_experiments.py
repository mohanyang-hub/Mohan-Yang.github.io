import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


SEED = 42
DATA_PATH = r"C:\Users\Che\Desktop\3.16\full_819_spectro_physio_data.csv"
FEATURE_ROOT = r"C:\Users\Che\Desktop\3.16"
OUTPUT_DIR = r"C:\Users\Che\Documents\Codex\2026-05-21\new-chat-2\amaf_lite_819_outputs"

EPOCHS = 30
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 5
GRADE_MAP = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}
GRADE_NAMES = ["G1", "G3", "G5", "G7", "G9"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    return {
        "OA": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MAE": mean_absolute_error(y_true, y_pred),
        "QWK": cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES))),
    }


def load_plot_metadata():
    rows = []
    for grade in [1, 3, 5, 7, 9]:
        path = os.path.join(FEATURE_ROOT, f"Level_{grade}", f"Labeled_Features_L{grade}.csv")
        df = pd.read_csv(path)
        rows.append(df[["Plot_ID", "Cultivar", "ID", "Level"]].copy())
    meta = pd.concat(rows, ignore_index=True)
    meta["Plot_ID"] = meta["Plot_ID"].astype(str)
    return meta


def load_819_data():
    df = pd.read_csv(DATA_PATH)
    meta = load_plot_metadata()
    if len(df) != len(meta):
        raise ValueError(f"Merged data has {len(df)} rows, but metadata has {len(meta)} rows.")

    df = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    if not (df["Disease_Grade"].astype(int).values == df["Level"].astype(int).values).all():
        raise ValueError("Disease_Grade in merged data does not match Level in 2D metadata.")

    spec_cols = sorted([c for c in df.columns if c.startswith("Wave_")], key=lambda x: int(x.split("_")[1]))
    hand_cols = [c for c in df.columns if c not in spec_cols + ["Disease_Grade", "SPAD", "Plot_ID", "Cultivar", "ID", "Level"]]

    X_spec = df[spec_cols].values.astype(np.float32)
    X_spad = df[["SPAD"]].values.astype(np.float32)
    X_hand = df[hand_cols].values.astype(np.float32)
    y = df["Disease_Grade"].astype(int).map(GRADE_MAP).values.astype(np.int64)
    plot_ids = df["Plot_ID"].values

    print("=" * 72)
    print("819-sample data loaded")
    print("=" * 72)
    print(f"Samples: {len(y)}")
    print(f"Spectral dim: {X_spec.shape[1]}")
    print(f"SPAD dim: {X_spad.shape[1]}")
    print(f"Handcrafted dim: {X_hand.shape[1]}")
    print(f"Class distribution: {np.bincount(y, minlength=NUM_CLASSES)}")
    print("Plot-grade distribution:")
    print(pd.crosstab(df["Plot_ID"], df["Disease_Grade"]))
    return df, X_spec, X_spad, X_hand, y, plot_ids, spec_cols, hand_cols


def class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


class SpectralEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=48):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.fc(x)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=48, hidden=96, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BottleneckFusion2(nn.Module):
    def __init__(self, dim=48, num_bottlenecks=2, num_heads=4, dropout=0.15):
        super().__init__()
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)
        self.type_m = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.type_b = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)
        self.b_to_m = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.m_to_b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_b = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, tokens):
        bsz = tokens.size(0)
        m = tokens + self.type_m
        b = self.bottlenecks.expand(bsz, -1, -1) + self.type_b
        b_update, attn = self.b_to_m(query=b, key=m, value=m, need_weights=True, average_attn_weights=True)
        b = self.norm_b(b + b_update)
        m_update, _ = self.m_to_b(query=m, key=b, value=b, need_weights=False)
        m = self.norm_m(m + m_update)
        b = b + self.ffn(b)
        weights = attn.mean(dim=1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return m, b.mean(dim=1), weights


class AMAF819(nn.Module):
    def __init__(self, spec_dim, hand_dim, mode="amaf"):
        super().__init__()
        self.mode = mode
        self.spec = SpectralEncoder(spec_dim, out_dim=48)
        self.spad = MLPEncoder(1, out_dim=8, hidden=16, dropout=0.1)
        self.spec_spad = nn.Sequential(nn.Linear(56, 48), nn.BatchNorm1d(48), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.hand = MLPEncoder(hand_dim, out_dim=48, hidden=96, dropout=0.25)
        self.bottleneck = BottleneckFusion2(dim=48)
        self.gate = nn.Sequential(nn.Linear(96, 48), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(48, 2))
        cls_in = 96 if mode == "concat" else 48
        self.cls = nn.Sequential(nn.Linear(cls_in, 48), nn.BatchNorm1d(48), nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(48, NUM_CLASSES))

    def forward(self, spec, spad, hand):
        z_spec = self.spec_spad(torch.cat([self.spec(spec), self.spad(spad)], dim=-1))
        z_hand = self.hand(hand)
        if self.mode == "concat":
            weights = torch.full((spec.size(0), 2), 0.5, device=spec.device)
            return self.cls(torch.cat([z_spec, z_hand], dim=-1)), weights
        if self.mode == "simple":
            logits = self.gate(torch.cat([z_spec, z_hand], dim=-1))
            weights = F.softmax(logits / 0.8, dim=-1)
            fused = weights[:, :1] * z_spec + weights[:, 1:] * z_hand
            return self.cls(fused), weights
        tokens, bottleneck_mean, weights_b = self.bottleneck(torch.stack([z_spec, z_hand], dim=1))
        logits = self.gate(torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1))
        weights = F.softmax(logits / 0.8, dim=-1)
        fused = weights[:, :1] * tokens[:, 0] + weights[:, 1:] * tokens[:, 1] + 0.2 * bottleneck_mean
        return self.cls(fused), weights


STAGE_PRIOR_2 = torch.tensor(
    [
        [0.65, 0.35],
        [0.60, 0.40],
        [0.48, 0.52],
        [0.38, 0.62],
        [0.30, 0.70],
    ],
    dtype=torch.float32,
)


def stage_loss(weights, labels):
    target = STAGE_PRIOR_2.to(weights.device)[labels]
    return F.kl_div(torch.log(weights.clamp_min(1e-8)), target, reduction="batchmean")


def fit_predict_neural(X_spec, X_spad, X_hand, y, train_idx, test_idx, mode, device, use_stage_prior):
    scalers = [StandardScaler(), StandardScaler(), StandardScaler()]
    s_tr = scalers[0].fit_transform(X_spec[train_idx])
    s_te = scalers[0].transform(X_spec[test_idx])
    sp_tr = scalers[1].fit_transform(X_spad[train_idx])
    sp_te = scalers[1].transform(X_spad[test_idx])
    h_tr = scalers[2].fit_transform(X_hand[train_idx])
    h_te = scalers[2].transform(X_hand[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    train_ds = TensorDataset(
        torch.tensor(s_tr, dtype=torch.float32),
        torch.tensor(sp_tr, dtype=torch.float32),
        torch.tensor(h_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = AMAF819(s_tr.shape[1], h_tr.shape[1], mode=mode).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    weights_cls = class_weights(y_tr).to(device)

    for epoch in range(EPOCHS):
        model.train()
        for spec, spad, hand, labels in train_loader:
            spec, spad, hand, labels = spec.to(device), spad.to(device), hand.to(device), labels.to(device)
            spec = spec + torch.randn_like(spec) * 0.01
            optimizer.zero_grad()
            logits, weights = model(spec, spad, hand)
            loss = F.cross_entropy(logits, labels, weight=weights_cls)
            if use_stage_prior:
                loss = loss + 0.03 * stage_loss(weights, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits, weights = model(
            torch.tensor(s_te, dtype=torch.float32).to(device),
            torch.tensor(sp_te, dtype=torch.float32).to(device),
            torch.tensor(h_te, dtype=torch.float32).to(device),
        )
    return y_te, logits.argmax(1).cpu().numpy(), weights.cpu().numpy()


def run_ml_baseline(name, estimator, X, y, plot_ids):
    fold_rows, yt_all, yp_all = [], [], []
    for fold, plot in enumerate(sorted(np.unique(plot_ids), key=lambda x: int(str(x).split("_")[-1])), start=1):
        train_idx = plot_ids != plot
        test_idx = plot_ids == plot
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        metrics = compute_metrics(y[test_idx], pred)
        fold_rows.append({"Experiment": name, "Fold": fold, "Test_Plot": plot, "n_test": int(test_idx.sum()), **metrics})
        yt_all.append(y[test_idx])
        yp_all.append(pred)
    return pd.DataFrame(fold_rows), np.concatenate(yt_all), np.concatenate(yp_all)


def run_neural_experiment(name, X_spec, X_spad, X_hand, y, plot_ids, mode, use_stage_prior, device):
    fold_rows, yt_all, yp_all, w_all = [], [], [], []
    for fold, plot in enumerate(sorted(np.unique(plot_ids), key=lambda x: int(str(x).split("_")[-1])), start=1):
        print(f"{name} | Fold {fold} | Test {plot}")
        train_idx = plot_ids != plot
        test_idx = plot_ids == plot
        yt, yp, weights = fit_predict_neural(X_spec, X_spad, X_hand, y, train_idx, test_idx, mode, device, use_stage_prior)
        metrics = compute_metrics(yt, yp)
        fold_rows.append({"Experiment": name, "Fold": fold, "Test_Plot": plot, "n_test": int(test_idx.sum()), **metrics})
        yt_all.append(yt)
        yp_all.append(yp)
        w_all.append(weights)
        print(f"  OA={metrics['OA']:.4f}, F1={metrics['Macro-F1']:.4f}, MAE={metrics['MAE']:.4f}, QWK={metrics['QWK']:.4f}")
    return pd.DataFrame(fold_rows), np.concatenate(yt_all), np.concatenate(yp_all), np.concatenate(w_all)


def summarize_fold_df(fold_df):
    return {
        "Experiment": fold_df["Experiment"].iloc[0],
        "OA": f"{fold_df['OA'].mean():.4f} ± {fold_df['OA'].std(ddof=0):.4f}",
        "Macro-F1": f"{fold_df['Macro-F1'].mean():.4f} ± {fold_df['Macro-F1'].std(ddof=0):.4f}",
        "MAE": f"{fold_df['MAE'].mean():.4f} ± {fold_df['MAE'].std(ddof=0):.4f}",
        "QWK": f"{fold_df['QWK'].mean():.4f} ± {fold_df['QWK'].std(ddof=0):.4f}",
        "Overall_OA": compute_metrics(fold_df.attrs["y_true"], fold_df.attrs["y_pred"])["OA"],
        "Overall_Macro-F1": compute_metrics(fold_df.attrs["y_true"], fold_df.attrs["y_pred"])["Macro-F1"],
        "Overall_MAE": compute_metrics(fold_df.attrs["y_true"], fold_df.attrs["y_pred"])["MAE"],
        "Overall_QWK": compute_metrics(fold_df.attrs["y_true"], fold_df.attrs["y_pred"])["QWK"],
    }


def main():
    set_seed()
    df, X_spec, X_spad, X_hand, y, plot_ids, spec_cols, hand_cols = load_819_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_no_spad = np.hstack([X_spec, X_hand])
    all_fold_dfs = []
    summaries = []

    ml_configs = [
        ("SVM_without_SPAD", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced"), X_no_spad),
        ("RF_without_SPAD", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=-1), X_no_spad),
        ("XGBoost_without_SPAD", XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.75, objective="multi:softprob", num_class=NUM_CLASSES, eval_metric="mlogloss", random_state=SEED, n_jobs=-1), X_no_spad),
    ]
    for name, estimator, X in ml_configs:
        print(f"\nRunning {name}")
        fold_df, yt, yp = run_ml_baseline(name, estimator, X, y, plot_ids)
        fold_df.attrs["y_true"], fold_df.attrs["y_pred"] = yt, yp
        all_fold_dfs.append(fold_df)
        summaries.append(summarize_fold_df(fold_df))

    neural_configs = [
        ("FeatureConcat_SpectralSPAD_Handcrafted", "concat", False),
        ("SimpleAttention_SpectralSPAD_Handcrafted", "simple", False),
        ("AMAF_lite_Bottleneck_StagePrior", "amaf", True),
    ]
    attention_outputs = []
    for name, mode, use_stage_prior in neural_configs:
        print(f"\nRunning {name}")
        fold_df, yt, yp, weights = run_neural_experiment(name, X_spec, X_spad, X_hand, y, plot_ids, mode, use_stage_prior, device)
        fold_df.attrs["y_true"], fold_df.attrs["y_pred"] = yt, yp
        all_fold_dfs.append(fold_df)
        summaries.append(summarize_fold_df(fold_df))
        attn = pd.DataFrame({"Experiment": name, "y_true": yt, "y_pred": yp, "attn_spectral_spad": weights[:, 0], "attn_handcrafted": weights[:, 1]})
        attention_outputs.append(attn)
        if name == "AMAF_lite_Bottleneck_StagePrior":
            cm = confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES)))
            pd.DataFrame(cm, index=GRADE_NAMES, columns=GRADE_NAMES).to_csv(os.path.join(OUTPUT_DIR, "amaf_lite_confusion_matrix.csv"), encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    for item in all_fold_dfs:
        item.attrs = {}
    fold_all = pd.concat(all_fold_dfs, ignore_index=True)
    attn_all = pd.concat(attention_outputs, ignore_index=True)

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_819_lopo.csv"), index=False, encoding="utf-8-sig")
    fold_all.to_csv(os.path.join(OUTPUT_DIR, "fold_metrics_819_lopo.csv"), index=False, encoding="utf-8-sig")
    attn_all.to_csv(os.path.join(OUTPUT_DIR, "attention_predictions_819.csv"), index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(OUTPUT_DIR, "full_819_with_plot_metadata.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("819 LOPO summary")
    print("=" * 72)
    print(summary_df[["Experiment", "OA", "Macro-F1", "MAE", "QWK", "Overall_OA", "Overall_Macro-F1", "Overall_MAE", "Overall_QWK"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
