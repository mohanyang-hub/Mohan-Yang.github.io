import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# =========================================================
# Paths and training config
# =========================================================
PATH_MERGED = r"C:\Users\Che\Desktop\5.23\全模态_最终总表.xlsx"
PATH_IMG_TABLE = r"C:\Users\Che\Desktop\3.16\175_2D_final\175_植被_纹理_图像_总表.xlsx"
IMAGE_ROOT = r"C:\Users\Che\Desktop\3.16\175_2D_final\Images"
OUTPUT_DIR = r"C:\Users\Che\Documents\Codex\2026-05-21\new-chat-2\adaptive_attention_outputs"

EPOCHS = 80                  # 正常训练轮数
ABLATION_EPOCHS = 50         # 消融实验训练轮数
BATCH_SIZE = 8               # 批次大小
LR = 1e-4                    # 学习率
WEIGHT_DECAY = 1e-4          # 权重衰减（正则化）
REPORT_MODE = "final"        # 报告模式：final=最后一轮，best_test=最优轮
RUN_ABLATION = False         # 是否运行消融实验
# Pure bottleneck fusion: set this to 0.0.
# Bottleneck + stage-aware prior: use 0.03-0.08.
ATTENTION_PRIOR_LAMBDA = 0.05
ATTENTION_TEMPERATURE = 0.7
FUSION_MODE = "bottleneck"  # "simple_attention" or "bottleneck"
BOTTLENECK_TOKENS = 2
BOTTLENECK_HEADS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 特征列名定义（手工植被指数+纹理+3D结构）
HAND_FEATURES = [
    "NDVI", "RCI", "PSSR", "SR", "CI_GREEN", "SR_GREEN", "GREDEDGE", "VI_REDED",
    "VIPA", "CARI", "SAVI", "MSAVI", "OSAVI", "CVI2", "ARI", "PBI", "LAI_EST",
    "RDVI", "DVI", "Contrast", "Correlation", "Energy", "Homogeneity",
]

STRUCTURE_FEATURES = [
    "PH", "Canopy_Width_X", "Canopy_Width_Y", "Canopy_Area", "Voxel_Volume", "Projection_Density",
]
# 病害等级映射
GRADE_MAP = {"G1": 0, "G3": 1, "G5": 2, "G7": 3, "G9": 4}
GRADE_NAMES = ["G1", "G3", "G5", "G7", "G9"]
NUM_CLASSES = 5
MODALITY_NAMES = ["Spectral_SPAD", "Handcrafted", "R_G_NIR", "Structure_3D"]
#4. 核心：Stage-aware Prior（阶段感知生物先验）
# 四模态权重：光谱SPAD、手工特征、图像、3D结构
# 含义：病害从轻到重，各模态重要性变化规律
# Stage-aware biological prior for attention weights.
# Order: Spectral/SPAD, Handcrafted VI/texture, R-G-NIR false-color image, 3D structure.
ATTENTION_PRIOR = torch.tensor(
    [
        [0.45, 0.25, 0.20, 0.10],  # G1: physiological/spectral signals dominate
        [0.40, 0.25, 0.23, 0.12],  # G3: early disease, red-edge/SPAD still important
        [0.28, 0.30, 0.27, 0.15],  # G5: mixed spectral, VI/texture and false-color canopy symptoms
        [0.18, 0.24, 0.24, 0.34],  # G7: structural degradation becomes important
        [0.12, 0.18, 0.20, 0.50],  # G9: canopy collapse and 3D traits dominate
    ],
    dtype=torch.float32,
)
#训练时约束模型的注意力权重，让它符合病害发展规律
# 5. 网络模块定义
# =========================================================

# ----------------------
# 图像编码器：CNN提取图像特征
class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, out_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, out_dim), nn.ReLU(inplace=True), nn.Dropout(0.3))

    def forward(self, x):
        return self.fc(self.conv(x))
# SE通道注意力（1D时序版）
# 作用：自动给重要特征通道加权
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
# 多尺度光谱残差块
class SpectralBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size, padding=padding),
            nn.BatchNorm1d(32),
        )
        self.shortcut = nn.Conv1d(1, 32, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

# 光谱编码器：多尺度卷积+SE
class SpectralEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.branch3 = SpectralBlock(3)
        self.branch5 = SpectralBlock(5)
        self.branch7 = SpectralBlock(7)
        self.se = SEBlock(96)
        self.pool = nn.AdaptiveAvgPool1d(64)
        # 用虚拟前向计算输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim)
            conv_out = self._forward_conv(dummy).flatten(1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
        )

    def _forward_conv(self, x):
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        return self.pool(self.se(x))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self._forward_conv(x)
        return self.mlp(x.flatten(1))


class SPADEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 8), nn.ReLU(inplace=True), nn.Linear(8, 8))

    def forward(self, x):
        return self.mlp(x)

# 通用MLP编码器（手工特征/3D结构
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)
# 模态注意力：学习4个模态的权重
class ModalityAttention(nn.Module):
    def __init__(self, temperature=0.7):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(32 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / self.temperature, dim=-1)

# 瓶颈注意力融合（创新点）
# 让4个模态通过少量瓶颈token交互，防止过拟合
class BottleneckAttentionFusion(nn.Module):
    """Lightweight bottleneck fusion for small-sample multimodal learning.

    Four modality embeddings first communicate only through a few learnable
    bottleneck tokens. This constrains cross-modal information exchange and
    outputs refined modality representations Z* for the stage-aware gating network.
    """

    def __init__(self, dim=32, num_modalities=4, num_bottlenecks=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)

        self.modality_type = nn.Parameter(torch.randn(1, num_modalities, dim) * 0.02)
        self.bottleneck_type = nn.Parameter(torch.randn(1, num_bottlenecks, dim) * 0.02)
        # 双向注意力：瓶颈 ↔ 模态
        self.b_to_m = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.m_to_b = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_m = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, modality_tokens):
        # modality_tokens: [B, 4, 32]
        bsz = modality_tokens.size(0)
        m = modality_tokens + self.modality_type
        b = self.bottlenecks.expand(bsz, -1, -1) + self.bottleneck_type
        # 瓶颈关注模态
        b_update, attn_b_to_m = self.b_to_m(
            query=b,
            key=m,
            value=m,
            need_weights=True,
            average_attn_weights=True,
        )
        b = self.norm_b(b + b_update)
        # 模态关注瓶颈
        m_update, _ = self.m_to_b(
            query=m,
            key=b,
            value=b,
            need_weights=False,
        )
        m = self.norm_m(m + m_update)
        b = b + self.ffn(b)

        refined_tokens = self.out(m)
        return refined_tokens, attn_b_to_m, b

#最终多模态融合网络
class AdvancedMFNet(nn.Module):
    def __init__(
        self,
        spec_dim,
        hand_dim,
        structure_dim,
        num_classes=5,
        fusion_mode=None,
        active_modalities=None,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode or FUSION_MODE
        self.active_modalities = active_modalities or {
            "spectral_spad": True,
            "handcrafted": True,
            "structure": True,
            "r_g_nir": True,
        }
        # 4个模态编码器
        self.spec = SpectralEncoder(spec_dim)
        self.spad = SPADEncoder()
        self.spec_fuse = nn.Sequential(
            nn.Linear(32 + 8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.hand = MLPEncoder(hand_dim, hidden_dim=64, out_dim=32, dropout=0.3)
        self.structure = MLPEncoder(structure_dim, hidden_dim=32, out_dim=32, dropout=0.3)
        self.img = ImageEncoder(input_channels=3, out_dim=32)
        # 注意力与融合
        self.modal_attn = ModalityAttention(temperature=ATTENTION_TEMPERATURE)
        self.stage_gate = ModalityAttention(temperature=ATTENTION_TEMPERATURE)
        self.concat_fuse = nn.Sequential(
            nn.Linear(32 * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
        )
        self.bottleneck_fusion = BottleneckAttentionFusion(
            dim=32,
            num_modalities=4,
            num_bottlenecks=BOTTLENECK_TOKENS,
            num_heads=BOTTLENECK_HEADS,
            dropout=0.2,
        )
        # 分类头
        self.cls = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes),
        )

    def forward(self, spec, spad, hand, structure, img):
        # 编码各模态
        f_spec = self.spec(spec)
        f_spad = self.spad(spad)
        f_spec_all = self.spec_fuse(torch.cat([f_spec, f_spad], dim=-1))
        f_hand = self.hand(hand)
        f_structure = self.structure(structure)
        f_img = self.img(img)
        # 消融实验：关闭某些模态（置0）
        if not self.active_modalities.get("spectral_spad", True):
            f_spec_all = torch.zeros_like(f_spec_all)
        if not self.active_modalities.get("handcrafted", True):
            f_hand = torch.zeros_like(f_hand)
        if not self.active_modalities.get("structure", True):
            f_structure = torch.zeros_like(f_structure)
        if not self.active_modalities.get("r_g_nir", True):
            f_img = torch.zeros_like(f_img)
        # 三种融合方式
        if self.fusion_mode == "feature_concat":
            fusion = self.concat_fuse(torch.cat([f_spec_all, f_hand, f_img, f_structure], dim=-1))
            active = torch.tensor(
                [
                    float(self.active_modalities.get("spectral_spad", True)),
                    float(self.active_modalities.get("handcrafted", True)),
                    float(self.active_modalities.get("r_g_nir", True)),
                    float(self.active_modalities.get("structure", True)),
                ],
                device=fusion.device,
            )
            weights = active.unsqueeze(0).expand(fusion.size(0), -1)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.fusion_mode == "simple_attention":
            fusion_all = torch.cat([f_spec_all, f_hand, f_img, f_structure], dim=-1)
            weights = self.modal_attn(fusion_all)
            fusion = (
                weights[:, 0:1] * f_spec_all
                + weights[:, 1:2] * f_hand
                + weights[:, 2:3] * f_img
                + weights[:, 3:4] * f_structure
            )
        elif self.fusion_mode == "bottleneck":
            modality_tokens = torch.stack([f_spec_all, f_hand, f_img, f_structure], dim=1)
            refined_tokens, _, _ = self.bottleneck_fusion(modality_tokens)
            weights = self.stage_gate(refined_tokens.flatten(1))
            fusion = (weights.unsqueeze(-1) * refined_tokens).sum(dim=1)
        else:
            raise ValueError("FUSION_MODE must be 'feature_concat', 'simple_attention' or 'bottleneck'.")

        return self.cls(fusion), weights
# 6. 数据集类：加载光谱、SPAD、手工特征、3D结构、图像
class MultiModalDataset(Dataset):
    def __init__(self, spec, spad, hand, structure, img_names, labels, img_root):
        self.spec = spec
        self.spad = spad
        self.hand = hand
        self.structure = structure
        self.img_names = img_names
        self.labels = labels
        self.img_root = img_root
        self.mode = "train"
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = torch.tensor(self.spec[idx], dtype=torch.float32)
        spad = torch.tensor(self.spad[idx], dtype=torch.float32)
        hand = torch.tensor(self.hand[idx], dtype=torch.float32)
        structure = torch.tensor(self.structure[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        # 训练时给光谱加噪声增强
        if self.mode == "train":
            spec = spec + torch.randn_like(spec) * 0.015

        img = Image.open(os.path.join(self.img_root, self.img_names[idx])).convert("RGB")
        img = self.transform(img)
        return spec, spad, hand, structure, img, y
#7.评估指标、损失函数、绘图工具
# 计算分类指标
def compute_metrics(y_true, y_pred):
    return {
        "OA": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MAE": mean_absolute_error(y_true, y_pred),
        "QWK": cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES))),
    }
# 类别不平衡权重
def class_weights_from_labels(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)
# 阶段感知先验损失（KL散度约束注意力权重）
def attention_prior_loss(weights, labels):
    target = ATTENTION_PRIOR.to(weights.device)[labels]
    return F.kl_div(torch.log(weights.clamp_min(1e-8)), target, reduction="batchmean")
# 统计每个病害等级的平均模态权重
def summarize_attention(weights, labels):
    rows = []
    for class_id, grade in enumerate(GRADE_NAMES):
        mask = labels == class_id
        row = {"Grade": grade, "n": int(mask.sum())}
        if mask.sum() == 0:
            row.update({name: np.nan for name in MODALITY_NAMES})
        else:
            mean_w = weights[mask].mean(axis=0)
            row.update({name: float(mean_w[i]) for i, name in enumerate(MODALITY_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows)

# 8. 数据加载
def load_all_data():
    df = pd.read_excel(PATH_MERGED)
    spec_cols = []
    for col in df.columns:
        if col.startswith("Wave_"):
            num = col.split("_")[1]
            if num.isdigit():
                spec_cols.append(col)
    spec_cols.sort(key=lambda x: int(x.split("_")[1]))

    hand_cols = [c for c in HAND_FEATURES if c in df.columns]
    structure_cols = [c for c in STRUCTURE_FEATURES if c in df.columns]

    X_spec = df[spec_cols].values.astype(np.float32)
    X_spad = df[["SPAD"]].values.astype(np.float32)
    X_hand = df[hand_cols].values.astype(np.float32)
    X_structure = df[structure_cols].values.astype(np.float32)
    y = df["Grade"].map(GRADE_MAP).values.astype(np.int64)
    plot_ids = df["Plot"].astype(str).str.extract(r"(\d+)")[0].astype(int).values

    df_img = pd.read_excel(PATH_IMG_TABLE)
    img_names = df_img["Image_Name"].values.astype(str)
    if len(img_names) != len(y):
        raise ValueError(f"Image table length {len(img_names)} does not match merged table length {len(y)}.")

    print("=" * 60)
    print("Data loaded")
    print("=" * 60)
    print(f"Samples: {len(y)}")
    print(f"Spectral dim: {X_spec.shape[1]}")
    print(f"SPAD dim: {X_spad.shape[1]}")
    print(f"Handcrafted dim: {X_hand.shape[1]}")
    print(f"3D structure dim: {X_structure.shape[1]}")
    print(f"Class distribution: {np.bincount(y, minlength=NUM_CLASSES)}")
    print(f"Plots: {np.unique(plot_ids)}")
    print(f"Fusion mode: {FUSION_MODE}")
    print(f"Attention prior lambda: {ATTENTION_PRIOR_LAMBDA}")
    return X_spec, X_spad, X_hand, X_structure, img_names, y, plot_ids

# 9. 绘图函数
def plot_training_curves(history_df):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for fold_id, g in history_df.groupby("Fold"):
        axes[0, 0].plot(g["Epoch"], g["Train_Loss"], label=f"Fold {fold_id}")
        axes[0, 1].plot(g["Epoch"], g["Train_OA"], label=f"Train {fold_id}")
        axes[0, 1].plot(g["Epoch"], g["Test_OA"], "--", label=f"Test {fold_id}")
        axes[1, 0].plot(g["Epoch"], g["Test_Macro-F1"], label=f"Fold {fold_id}")
        axes[1, 1].plot(g["Epoch"], g["Test_QWK"], label=f"Fold {fold_id}")
    titles = ["Training Loss", "Train and Test OA", "Test Macro-F1", "Test QWK"]
    ylabels = ["Loss", "OA", "Macro-F1", "QWK"]
    for ax, title, ylabel in zip(axes.ravel(), titles, ylabels):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "training_curves_metrics.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_attention(attention_df):
    plt.rcParams["font.family"] = "Times New Roman"
    ax = attention_df.set_index("Grade")[MODALITY_NAMES].plot(kind="bar", figsize=(8, 5), width=0.75)
    ax.set_xlabel("Disease Severity Grade")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_ylim(0, 1)
    ax.set_title("Stage-aware Modality Weights")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attention_by_grade.png"), dpi=600, bbox_inches="tight")
    plt.close()


def plot_confusion(cm):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(GRADE_NAMES)
    ax.set_yticklabels(GRADE_NAMES)
    ax.set_xlabel("Predicted Grade")
    ax.set_ylabel("True Grade")
    ax.set_title("Overall Confusion Matrix")
    max_val = cm.max() if cm.size else 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > max_val / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_confusion_matrix.png"), dpi=600, bbox_inches="tight")
    plt.close()

# 10. 消融实验配置
ABLATION_CONFIGS = [
    {
        "name": "Full_Bottleneck",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": True, "handcrafted": True, "structure": True, "r_g_nir": True},
        "attn_lambda": 0.05,
    },
    {
        "name": "Full_SimpleAttention",
        "fusion_mode": "simple_attention",
        "active": {"spectral_spad": True, "handcrafted": True, "structure": True, "r_g_nir": True},
        "attn_lambda": 0.05,
    },
    {
        "name": "No_SpectralSPAD",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": False, "handcrafted": True, "structure": True, "r_g_nir": True},
        "attn_lambda": 0.0,
    },
    {
        "name": "No_Handcrafted",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": True, "handcrafted": False, "structure": True, "r_g_nir": True},
        "attn_lambda": 0.0,
    },
    {
        "name": "No_3D",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": True, "handcrafted": True, "structure": False, "r_g_nir": True},
        "attn_lambda": 0.0,
    },
    {
        "name": "No_R_G_NIR",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": True, "handcrafted": True, "structure": True, "r_g_nir": False},
        "attn_lambda": 0.0,
    },
    {
        "name": "SpectralSPAD_Only",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": True, "handcrafted": False, "structure": False, "r_g_nir": False},
        "attn_lambda": 0.0,
    },
    {
        "name": "Handcrafted_Only",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": False, "handcrafted": True, "structure": False, "r_g_nir": False},
        "attn_lambda": 0.0,
    },
    {
        "name": "Structure3D_Only",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": False, "handcrafted": False, "structure": True, "r_g_nir": False},
        "attn_lambda": 0.0,
    },
    {
        "name": "R_G_NIR_Only",
        "fusion_mode": "bottleneck",
        "active": {"spectral_spad": False, "handcrafted": False, "structure": False, "r_g_nir": True},
        "attn_lambda": 0.0,
    },
]
# 11. LOPO留一交叉验证实验主逻辑

def run_lopo_experiment(
    X_spec,
    X_spad,
    X_hand,
    X_structure,
    img_names,
    y,
    plot_ids,
    device,
    config,
    epochs,
    save_outputs=False,
):
    fold_rows = []
    history_rows = []
    overall_true, overall_pred, overall_weights = [], [], []

    for fold, test_plot in enumerate(np.unique(plot_ids), start=1):
        print(f"\n[{config['name']}] Fold {fold} | Test plot: {test_plot}")
        train_idx = plot_ids != test_plot
        test_idx = plot_ids == test_plot

        s_tr, s_te = X_spec[train_idx], X_spec[test_idx]
        sp_tr, sp_te = X_spad[train_idx], X_spad[test_idx]
        h_tr, h_te = X_hand[train_idx], X_hand[test_idx]
        st_tr, st_te = X_structure[train_idx], X_structure[test_idx]
        img_tr, img_te = img_names[train_idx], img_names[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        ss, ssp, sh, stc = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
        s_tr, s_te = ss.fit_transform(s_tr), ss.transform(s_te)
        sp_tr, sp_te = ssp.fit_transform(sp_tr), ssp.transform(sp_te)
        h_tr, h_te = sh.fit_transform(h_tr), sh.transform(h_te)
        st_tr, st_te = stc.fit_transform(st_tr), stc.transform(st_te)

        train_ds = MultiModalDataset(s_tr, sp_tr, h_tr, st_tr, img_tr, y_tr, IMAGE_ROOT)
        test_ds = MultiModalDataset(s_te, sp_te, h_te, st_te, img_te, y_te, IMAGE_ROOT)
        train_ds.mode = "train"
        test_ds.mode = "test"
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = AdvancedMFNet(
            s_tr.shape[1],
            h_tr.shape[1],
            st_tr.shape[1],
            NUM_CLASSES,
            fusion_mode=config["fusion_mode"],
            active_modalities=config["active"],
        ).to(device)
        cls_weights = class_weights_from_labels(y_tr).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)

        best_score, best_epoch = -1, 0
        best_metrics = best_pred = best_true = best_weights = None
        final_metrics = final_pred = final_true = final_weights = None

        for epoch in range(1, epochs + 1):
            model.train()
            train_preds, train_trues = [], []
            total_loss, total_cls_loss, total_attn_loss, total_count = 0.0, 0.0, 0.0, 0

            for s, sp, h, st, img, yy in train_loader:
                s, sp, h, st, img, yy = (
                    s.to(device),
                    sp.to(device),
                    h.to(device),
                    st.to(device),
                    img.to(device),
                    yy.to(device),
                )
                optimizer.zero_grad()
                out, attn = model(s, sp, h, st, img)
                loss_cls = F.cross_entropy(out, yy, weight=cls_weights)
                lambda_base = config.get("attn_lambda", 0.0)
                warmup_epochs = config.get("stage_warmup_epochs", 0)
                if warmup_epochs > 0 and lambda_base > 0:
                    if epoch <= warmup_epochs:
                        lambda_stage = 0.0
                    else:
                        lambda_stage = lambda_base * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
                else:
                    lambda_stage = lambda_base

                loss_attn = (
                    attention_prior_loss(attn, yy)
                    if lambda_stage > 0
                    else torch.tensor(0.0, device=device)
                )
                loss = loss_cls + lambda_stage * loss_attn
                loss.backward()
                optimizer.step()

                pred = out.argmax(1)
                total_loss += loss.item() * yy.size(0)
                total_cls_loss += loss_cls.item() * yy.size(0)
                total_attn_loss += loss_attn.item() * yy.size(0)
                total_count += yy.size(0)
                train_preds.extend(pred.detach().cpu().numpy())
                train_trues.extend(yy.detach().cpu().numpy())

            scheduler.step()
            train_metrics = compute_metrics(train_trues, train_preds)

            model.eval()
            test_preds, test_trues, test_weights = [], [], []
            with torch.no_grad():
                for s, sp, h, st, img, yy in test_loader:
                    s, sp, h, st, img = s.to(device), sp.to(device), h.to(device), st.to(device), img.to(device)
                    out, attn = model(s, sp, h, st, img)
                    test_preds.extend(out.argmax(1).cpu().numpy())
                    test_trues.extend(yy.numpy())
                    test_weights.append(attn.cpu().numpy())

            test_weights = np.concatenate(test_weights, axis=0)
            test_metrics = compute_metrics(test_trues, test_preds)

            if test_metrics["Macro-F1"] > best_score:
                best_score = test_metrics["Macro-F1"]
                best_epoch = epoch
                best_metrics = test_metrics
                best_pred = np.array(test_preds)
                best_true = np.array(test_trues)
                best_weights = test_weights

            final_metrics = test_metrics
            final_pred = np.array(test_preds)
            final_true = np.array(test_trues)
            final_weights = test_weights

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d} | "
                    f"TrainOA={train_metrics['OA']:.4f} | "
                    f"TestOA={test_metrics['OA']:.4f} | "
                    f"F1={test_metrics['Macro-F1']:.4f} | "
                    f"MAE={test_metrics['MAE']:.4f} | "
                    f"QWK={test_metrics['QWK']:.4f}"
                )

            history_rows.append(
                {
                    "Experiment": config["name"],
                    "Fold": fold,
                    "Test_Plot": test_plot,
                    "Epoch": epoch,
                    "Train_Loss": total_loss / max(total_count, 1),
                    "Train_Cls_Loss": total_cls_loss / max(total_count, 1),
                    "Train_Attn_Loss": total_attn_loss / max(total_count, 1),
                    "Train_OA": train_metrics["OA"],
                    "Train_Macro-F1": train_metrics["Macro-F1"],
                    "Train_MAE": train_metrics["MAE"],
                    "Train_QWK": train_metrics["QWK"],
                    "Test_OA": test_metrics["OA"],
                    "Test_Macro-F1": test_metrics["Macro-F1"],
                    "Test_MAE": test_metrics["MAE"],
                    "Test_QWK": test_metrics["QWK"],
                }
            )

        if REPORT_MODE == "best_test":
            report_epoch, report_metrics, report_pred, report_true, report_weights = (
                best_epoch,
                best_metrics,
                best_pred,
                best_true,
                best_weights,
            )
        else:
            report_epoch, report_metrics, report_pred, report_true, report_weights = (
                epochs,
                final_metrics,
                final_pred,
                final_true,
                final_weights,
            )

        fold_rows.append(
            {
                "Experiment": config["name"],
                "Fusion_Mode": config["fusion_mode"],
                "Active_Modalities": ",".join(k for k, v in config["active"].items() if v),
                "Attention_Lambda": config.get("attn_lambda", 0.0),
                "Fold": fold,
                "Test_Plot": test_plot,
                "Report_Mode": REPORT_MODE,
                "Report_Epoch": report_epoch,
                "Best_Test_Epoch": best_epoch,
                "Best_Test_Macro-F1": best_score,
                "n_test": int(len(y_te)),
                "OA": report_metrics["OA"],
                "Macro-F1": report_metrics["Macro-F1"],
                "MAE": report_metrics["MAE"],
                "QWK": report_metrics["QWK"],
            }
        )
        overall_true.append(report_true)
        overall_pred.append(report_pred)
        overall_weights.append(report_weights)

    history_df = pd.DataFrame(history_rows)
    fold_df = pd.DataFrame(fold_rows)
    all_true = np.concatenate(overall_true)
    all_pred = np.concatenate(overall_pred)
    all_weights = np.concatenate(overall_weights)
    overall_metrics = compute_metrics(all_true, all_pred)
    attention_df = summarize_attention(all_weights, all_true)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))

    if save_outputs:
        history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history_by_epoch.csv"), index=False, encoding="utf-8-sig")
        fold_df.to_csv(os.path.join(OUTPUT_DIR, "lopo_fold_metrics.csv"), index=False, encoding="utf-8-sig")
        attention_df.to_csv(os.path.join(OUTPUT_DIR, "attention_by_grade.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(cm, index=GRADE_NAMES, columns=GRADE_NAMES).to_csv(
            os.path.join(OUTPUT_DIR, "overall_confusion_matrix.csv"), encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "y_true": all_true,
                "y_pred": all_pred,
                "true_grade": [GRADE_NAMES[i] for i in all_true],
                "pred_grade": [GRADE_NAMES[i] for i in all_pred],
                "attn_spectral_spad": all_weights[:, 0],
                "attn_handcrafted": all_weights[:, 1],
                "attn_r_g_nir": all_weights[:, 2],
                "attn_structure_3d": all_weights[:, 3],
            }
        ).to_csv(os.path.join(OUTPUT_DIR, "lopo_predictions.csv"), index=False, encoding="utf-8-sig")
        plot_training_curves(history_df)
        plot_attention(attention_df)
        plot_confusion(cm)

    summary = {
        "Experiment": config["name"],
        "Fusion_Mode": config["fusion_mode"],
        "Active_Modalities": ",".join(k for k, v in config["active"].items() if v),
        "Attention_Lambda": config.get("attn_lambda", 0.0),
        "Fold_OA_Mean": fold_df["OA"].mean(),
        "Fold_OA_Std": fold_df["OA"].std(ddof=0),
        "Fold_MacroF1_Mean": fold_df["Macro-F1"].mean(),
        "Fold_MacroF1_Std": fold_df["Macro-F1"].std(ddof=0),
        "Fold_MAE_Mean": fold_df["MAE"].mean(),
        "Fold_MAE_Std": fold_df["MAE"].std(ddof=0),
        "Fold_QWK_Mean": fold_df["QWK"].mean(),
        "Fold_QWK_Std": fold_df["QWK"].std(ddof=0),
        "Overall_OA": overall_metrics["OA"],
        "Overall_Macro-F1": overall_metrics["Macro-F1"],
        "Overall_MAE": overall_metrics["MAE"],
        "Overall_QWK": overall_metrics["QWK"],
    }
    return summary, fold_df, history_df, attention_df, cm

# 12. 主函数
def main():
    X_spec, X_spad, X_hand, X_structure, img_names, y, plot_ids = load_all_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if RUN_ABLATION:
        summaries = []
        all_folds = []
        all_attention = []
        for cfg in ABLATION_CONFIGS:
            print("\n" + "=" * 72)
            print(f"Running ablation: {cfg['name']}")
            print("=" * 72)
            summary, fold_df, history_df, attention_df, cm = run_lopo_experiment(
                X_spec,
                X_spad,
                X_hand,
                X_structure,
                img_names,
                y,
                plot_ids,
                device,
                cfg,
                epochs=ABLATION_EPOCHS,
                save_outputs=(cfg["name"] == "Full_Bottleneck"),
            )
            summaries.append(summary)
            all_folds.append(fold_df)
            attention_df.insert(0, "Experiment", cfg["name"])
            all_attention.append(attention_df)
            print(
                f"{cfg['name']} | "
                f"OA={summary['Fold_OA_Mean']:.4f}±{summary['Fold_OA_Std']:.4f} | "
                f"F1={summary['Fold_MacroF1_Mean']:.4f}±{summary['Fold_MacroF1_Std']:.4f} | "
                f"MAE={summary['Fold_MAE_Mean']:.4f} | "
                f"QWK={summary['Fold_QWK_Mean']:.4f}"
            )

        summary_df = pd.DataFrame(summaries).sort_values("Fold_MacroF1_Mean", ascending=False)
        folds_df = pd.concat(all_folds, ignore_index=True)
        attention_all_df = pd.concat(all_attention, ignore_index=True)
        summary_path = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
        folds_path = os.path.join(OUTPUT_DIR, "ablation_fold_metrics.csv")
        attention_path = os.path.join(OUTPUT_DIR, "ablation_attention_by_grade.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        folds_df.to_csv(folds_path, index=False, encoding="utf-8-sig")
        attention_all_df.to_csv(attention_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 72)
        print("Ablation summary")
        print("=" * 72)
        cols = [
            "Experiment",
            "Fold_OA_Mean",
            "Fold_OA_Std",
            "Fold_MacroF1_Mean",
            "Fold_MacroF1_Std",
            "Fold_MAE_Mean",
            "Fold_QWK_Mean",
        ]
        print(summary_df[cols].to_string(index=False))
        print("\nSaved:")
        print(summary_path)
        print(folds_path)
        print(attention_path)
        return

    fold_rows, history_rows = [], []
    overall_true, overall_pred, overall_weights = [], [], []

    for fold, test_plot in enumerate(np.unique(plot_ids), start=1):
        print(f"\nFold {fold} | Test plot: {test_plot}")
        train_idx = plot_ids != test_plot
        test_idx = plot_ids == test_plot

        s_tr, s_te = X_spec[train_idx], X_spec[test_idx]
        sp_tr, sp_te = X_spad[train_idx], X_spad[test_idx]
        h_tr, h_te = X_hand[train_idx], X_hand[test_idx]
        st_tr, st_te = X_structure[train_idx], X_structure[test_idx]
        img_tr, img_te = img_names[train_idx], img_names[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        ss, ssp, sh, stc = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
        s_tr, s_te = ss.fit_transform(s_tr), ss.transform(s_te)
        sp_tr, sp_te = ssp.fit_transform(sp_tr), ssp.transform(sp_te)
        h_tr, h_te = sh.fit_transform(h_tr), sh.transform(h_te)
        st_tr, st_te = stc.fit_transform(st_tr), stc.transform(st_te)

        train_ds = MultiModalDataset(s_tr, sp_tr, h_tr, st_tr, img_tr, y_tr, IMAGE_ROOT)
        test_ds = MultiModalDataset(s_te, sp_te, h_te, st_te, img_te, y_te, IMAGE_ROOT)
        train_ds.mode = "train"
        test_ds.mode = "test"
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = AdvancedMFNet(s_tr.shape[1], h_tr.shape[1], st_tr.shape[1], NUM_CLASSES).to(device)
        cls_weights = class_weights_from_labels(y_tr).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)

        best_score, best_epoch = -1, 0
        best_metrics = best_pred = best_true = best_weights = None
        final_metrics = final_pred = final_true = final_weights = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_preds, train_trues = [], []
            total_loss, total_cls_loss, total_attn_loss, total_count = 0.0, 0.0, 0.0, 0

            for s, sp, h, st, img, yy in train_loader:
                s, sp, h, st, img, yy = s.to(device), sp.to(device), h.to(device), st.to(device), img.to(device), yy.to(device)
                optimizer.zero_grad()
                out, attn = model(s, sp, h, st, img)
                loss_cls = F.cross_entropy(out, yy, weight=cls_weights)
                loss_attn = attention_prior_loss(attn, yy) if ATTENTION_PRIOR_LAMBDA > 0 else torch.tensor(0.0, device=device)
                loss = loss_cls + ATTENTION_PRIOR_LAMBDA * loss_attn
                loss.backward()
                optimizer.step()

                pred = out.argmax(1)
                total_loss += loss.item() * yy.size(0)
                total_cls_loss += loss_cls.item() * yy.size(0)
                total_attn_loss += loss_attn.item() * yy.size(0)
                total_count += yy.size(0)
                train_preds.extend(pred.detach().cpu().numpy())
                train_trues.extend(yy.detach().cpu().numpy())

            scheduler.step()
            train_metrics = compute_metrics(train_trues, train_preds)

            model.eval()
            test_preds, test_trues, test_weights = [], [], []
            with torch.no_grad():
                for s, sp, h, st, img, yy in test_loader:
                    s, sp, h, st, img = s.to(device), sp.to(device), h.to(device), st.to(device), img.to(device)
                    out, attn = model(s, sp, h, st, img)
                    test_preds.extend(out.argmax(1).cpu().numpy())
                    test_trues.extend(yy.numpy())
                    test_weights.append(attn.cpu().numpy())

            test_weights = np.concatenate(test_weights, axis=0)
            test_metrics = compute_metrics(test_trues, test_preds)

            if test_metrics["Macro-F1"] > best_score:
                best_score = test_metrics["Macro-F1"]
                best_epoch = epoch
                best_metrics = test_metrics
                best_pred = np.array(test_preds)
                best_true = np.array(test_trues)
                best_weights = test_weights

            final_metrics = test_metrics
            final_pred = np.array(test_preds)
            final_true = np.array(test_trues)
            final_weights = test_weights

            history_rows.append(
                {
                    "Fold": fold,
                    "Test_Plot": test_plot,
                    "Epoch": epoch,
                    "Train_Loss": total_loss / max(total_count, 1),
                    "Train_Cls_Loss": total_cls_loss / max(total_count, 1),
                    "Train_Attn_Loss": total_attn_loss / max(total_count, 1),
                    "Train_OA": train_metrics["OA"],
                    "Train_Macro-F1": train_metrics["Macro-F1"],
                    "Train_MAE": train_metrics["MAE"],
                    "Train_QWK": train_metrics["QWK"],
                    "Test_OA": test_metrics["OA"],
                    "Test_Macro-F1": test_metrics["Macro-F1"],
                    "Test_MAE": test_metrics["MAE"],
                    "Test_QWK": test_metrics["QWK"],
                }
            )

            print(
                f"Epoch {epoch:03d} | Loss={total_loss / max(total_count, 1):.4f} | "
                f"AttnLoss={total_attn_loss / max(total_count, 1):.4f} | "
                f"TrainOA={train_metrics['OA']:.4f} | TestOA={test_metrics['OA']:.4f} | "
                f"F1={test_metrics['Macro-F1']:.4f} | MAE={test_metrics['MAE']:.4f} | QWK={test_metrics['QWK']:.4f}"
            )

        if REPORT_MODE == "best_test":
            report_epoch, report_metrics, report_pred, report_true, report_weights = best_epoch, best_metrics, best_pred, best_true, best_weights
        else:
            report_epoch, report_metrics, report_pred, report_true, report_weights = EPOCHS, final_metrics, final_pred, final_true, final_weights

        fold_rows.append(
            {
                "Fold": fold,
                "Test_Plot": test_plot,
                "Report_Mode": REPORT_MODE,
                "Report_Epoch": report_epoch,
                "Best_Test_Epoch": best_epoch,
                "Best_Test_Macro-F1": best_score,
                "n_test": int(len(y_te)),
                "OA": report_metrics["OA"],
                "Macro-F1": report_metrics["Macro-F1"],
                "MAE": report_metrics["MAE"],
                "QWK": report_metrics["QWK"],
            }
        )
        overall_true.append(report_true)
        overall_pred.append(report_pred)
        overall_weights.append(report_weights)

    history_df = pd.DataFrame(history_rows)
    fold_df = pd.DataFrame(fold_rows)
    all_true = np.concatenate(overall_true)
    all_pred = np.concatenate(overall_pred)
    all_weights = np.concatenate(overall_weights)

    overall_metrics = compute_metrics(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    attention_df = summarize_attention(all_weights, all_true)

    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history_by_epoch.csv"), index=False, encoding="utf-8-sig")
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "lopo_fold_metrics.csv"), index=False, encoding="utf-8-sig")
    attention_df.to_csv(os.path.join(OUTPUT_DIR, "attention_by_grade.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cm, index=GRADE_NAMES, columns=GRADE_NAMES).to_csv(
        os.path.join(OUTPUT_DIR, "overall_confusion_matrix.csv"), encoding="utf-8-sig"
    )

    pd.DataFrame(
        {
            "y_true": all_true,
            "y_pred": all_pred,
            "true_grade": [GRADE_NAMES[i] for i in all_true],
            "pred_grade": [GRADE_NAMES[i] for i in all_pred],
            "attn_spectral_spad": all_weights[:, 0],
            "attn_handcrafted": all_weights[:, 1],
            "attn_r_g_nir": all_weights[:, 2],
            "attn_structure_3d": all_weights[:, 3],
        }
    ).to_csv(os.path.join(OUTPUT_DIR, "lopo_predictions.csv"), index=False, encoding="utf-8-sig")

    plot_training_curves(history_df)
    plot_attention(attention_df)
    plot_confusion(cm)

    print("\n" + "=" * 60)
    print("Final LOPO results")
    print("=" * 60)
    print(f"Report mode: {REPORT_MODE}")
    print(f"Mean OA: {fold_df['OA'].mean():.4f} +/- {fold_df['OA'].std(ddof=0):.4f}")
    print(f"Mean Macro-F1: {fold_df['Macro-F1'].mean():.4f} +/- {fold_df['Macro-F1'].std(ddof=0):.4f}")
    print(f"Mean MAE: {fold_df['MAE'].mean():.4f} +/- {fold_df['MAE'].std(ddof=0):.4f}")
    print(f"Mean QWK: {fold_df['QWK'].mean():.4f} +/- {fold_df['QWK'].std(ddof=0):.4f}")
    print("\nOverall pooled predictions:")
    print(f"OA: {overall_metrics['OA']:.4f}")
    print(f"Macro-F1: {overall_metrics['Macro-F1']:.4f}")
    print(f"MAE: {overall_metrics['MAE']:.4f}")
    print(f"QWK: {overall_metrics['QWK']:.4f}")
    print("\nAttention by grade:")
    print(attention_df)


if __name__ == "__main__":
    main()
