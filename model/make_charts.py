import os, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(os.path.dirname(HERE), "assets")
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

with open(os.path.join(HERE, "model_results.json")) as f:
    res = json.load(f)

COLORS = {
    "draft_position_only": "#9aa5b1",
    "logistic_regression": "#4c6ef5",
    "gradient_boosting": "#f76707",
    "deep_learning_mlp": "#12b886",
}
LABELS = {
    "draft_position_only": "Draft position only (status quo)",
    "logistic_regression": "Logistic regression (+ combine)",
    "gradient_boosting": "Gradient boosted trees",
    "deep_learning_mlp": "Deep learning (MLP)",
}

# --- ROC curve comparison ---
fig, ax = plt.subplots(figsize=(7, 6))
for name, d in res["roc"].items():
    auc = res["metrics"][name]["auc"]
    ax.plot(d["fpr"], d["tpr"], label=f"{LABELS[name]} (AUC={auc:.3f})", color=COLORS[name], linewidth=2.5)
ax.plot([0, 1], [0, 1], linestyle="--", color="#c0c0c0", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Predicting \"rotation-or-better\" NBA careers\nModel comparison on held-out draftees (2000\u20132016)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "roc_comparison.png"), dpi=160)
plt.close(fig)

# --- AUC bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
names = ["draft_position_only", "logistic_regression", "gradient_boosting", "deep_learning_mlp"]
aucs = [res["metrics"][n]["auc"] for n in names]
bars = ax.bar([LABELS[n] for n in names], aucs, color=[COLORS[n] for n in names])
for b, v in zip(bars, aucs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0.5, 0.95)
ax.set_ylabel("AUC (higher = better)")
ax.set_title("Combine measurements add real signal\nbeyond draft slot alone", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "auc_comparison.png"), dpi=160)
plt.close(fig)

# --- Feature importance ---
feat = pd.Series(res["top_features"]).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#e03131" if v < 0 else "#2f9e44" for v in feat.values]
ax.barh(feat.index.str.replace("_", " ").str.title(), feat.values, color=colors)
ax.axvline(0, color="#333", linewidth=0.8)
ax.set_xlabel("Standardized coefficient (logistic regression)")
ax.set_title("What predicts a rotation-caliber NBA career?", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=160)
plt.close(fig)

print("charts saved")
