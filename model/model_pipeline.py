import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

# All inputs are small, pre-extracted CSVs checked into model/data/ so this script
# runs standalone without needing the full 2.2GB nba.sqlite database. See
# build_dataset.py if you want to regenerate career_outcomes.csv from scratch
# against the raw Kaggle database.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = HERE

dc = pd.read_csv(os.path.join(DATA_DIR, "combine_clean.csv"))
career = pd.read_csv(os.path.join(DATA_DIR, "career_outcomes.csv"))
dh = pd.read_csv(os.path.join(DATA_DIR, "draft_history_clean.csv"))
dh["player_id"] = pd.to_numeric(dh["player_id"], errors="coerce")
dh["overall_pick"] = pd.to_numeric(dh["overall_pick"], errors="coerce")
dh = dh.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"], keep="first")

df = dc.merge(career, on="player_id", how="left")
df["games_played"] = df["games_played"].fillna(0)
df = df.merge(dh, on="player_id", how="left")

# target: became at least a rotation player (>=100 career games)
df["rotation_or_better"] = (df["games_played"] >= 100).astype(int)
df["career_years"] = pd.NA
mask = df["first_game_date"].notna()
df.loc[mask, "career_years"] = (
    pd.to_datetime(df.loc[mask, "last_game_date"]).dt.year
    - pd.to_datetime(df.loc[mask, "first_game_date"]).dt.year + 1
)

FEATURES = [
    "height_wo_shoes", "weight", "wingspan", "standing_reach", "body_fat_pct",
    "standing_vertical_leap", "max_vertical_leap", "lane_agility_time",
    "three_quarter_sprint", "bench_press",
]
DRAFT_FEATURE = ["overall_pick"]

# training universe: mature careers only (combine season <=2016 -> ~7+ yrs of outcome data by 2023 cutoff)
train_universe = df[df["season"] <= 2016].copy()
prospects = df[df["season"] >= 2019].copy()  # recent, still-developing players for demo

print("training universe:", train_universe.shape, "positive rate:", train_universe["rotation_or_better"].mean().round(3))
print("prospect demo set:", prospects.shape)

def prep_X(frame, cols):
    X = frame[cols].apply(pd.to_numeric, errors="coerce")
    return X

X_combine = prep_X(train_universe, FEATURES + DRAFT_FEATURE)
X_pickonly = prep_X(train_universe, DRAFT_FEATURE)
y = train_universe["rotation_or_better"].values

Xc_tr, Xc_te, Xp_tr, Xp_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_combine, X_pickonly, y, train_universe.index, test_size=0.25, random_state=42, stratify=y
)

imp_c = SimpleImputer(strategy="median").fit(Xc_tr)
scaler_c = StandardScaler().fit(imp_c.transform(Xc_tr))
Xc_tr_s = scaler_c.transform(imp_c.transform(Xc_tr))
Xc_te_s = scaler_c.transform(imp_c.transform(Xc_te))

imp_p = SimpleImputer(strategy="median").fit(Xp_tr)
Xp_tr_i = imp_p.transform(Xp_tr)
Xp_te_i = imp_p.transform(Xp_te)

results = {}

# --- Option A: draft position only (naive heuristic teams already use) ---
lr_pick = LogisticRegression(max_iter=1000).fit(Xp_tr_i, y_tr)
proba_pick = lr_pick.predict_proba(Xp_te_i)[:, 1]
results["draft_position_only"] = {
    "auc": roc_auc_score(y_te, proba_pick),
    "brier": brier_score_loss(y_te, proba_pick),
}

# --- Option B: interpretable baseline (logistic regression) w/ combine + pick ---
lr = LogisticRegression(max_iter=1000).fit(Xc_tr_s, y_tr)
proba_lr = lr.predict_proba(Xc_te_s)[:, 1]
results["logistic_regression"] = {
    "auc": roc_auc_score(y_te, proba_lr),
    "brier": brier_score_loss(y_te, proba_lr),
}
coef = pd.Series(lr.coef_[0], index=FEATURES + DRAFT_FEATURE).sort_values(key=abs, ascending=False)
coef_order = FEATURES + DRAFT_FEATURE  # column order matching Xc_*_s arrays
coef_vec = coef.reindex(coef_order).values

FEATURE_LABELS = {
    "height_wo_shoes": "Height (no shoes)",
    "weight": "Weight",
    "wingspan": "Wingspan",
    "standing_reach": "Standing reach",
    "body_fat_pct": "Body fat %",
    "standing_vertical_leap": "Standing vertical leap",
    "max_vertical_leap": "Max vertical leap",
    "lane_agility_time": "Lane agility time",
    "three_quarter_sprint": "Three-quarter sprint",
    "bench_press": "Bench press reps",
    "overall_pick": "Draft position",
}


# feature-specific phrasing so the raw z-score direction reads naturally
# (e.g. a fast sprint time is a LOW raw value, so "below average" would read
# as a weakness when it's actually a strength -- phrase per feature instead).
GENERIC_PHRASES = {
    2: "far above average", 1: "above average", 0: "about average",
    -1: "below average", -2: "far below average",
}
SPRINT_PHRASES = {
    2: "much slower than typical", 1: "slower than typical", 0: "typical speed",
    -1: "faster than typical", -2: "much faster than typical",
}
LEANNESS_PHRASES = {
    2: "higher body fat than typical", 1: "slightly higher body fat", 0: "typical body fat",
    -1: "leaner than typical", -2: "much leaner than typical",
}
PICK_PHRASES = {
    2: "very late pick", 1: "later pick", 0: "mid-round pick",
    -1: "earlier pick", -2: "very early pick",
}
FEATURE_PHRASES = {
    "lane_agility_time": SPRINT_PHRASES,
    "three_quarter_sprint": SPRINT_PHRASES,
    "body_fat_pct": LEANNESS_PHRASES,
    "overall_pick": PICK_PHRASES,
}


def _bucket(z):
    if z >= 1.5:
        return 2
    if z >= 0.5:
        return 1
    if z <= -1.5:
        return -2
    if z <= -0.5:
        return -1
    return 0


def magnitude_label(z, feature=None):
    phrases = FEATURE_PHRASES.get(feature, GENERIC_PHRASES)
    return phrases[_bucket(z)]


def explain_rows(Z, top_n=5):
    """Z: standardized feature matrix (n_rows x n_cols), columns aligned to coef_order.
    Returns a list (one per row) of the top_n factors driving that row's logistic-regression
    score, used as a plain-language 'why' explanation alongside the gradient-boosting grade.
    """
    contrib = Z * coef_vec  # elementwise, broadcasts coef_vec across rows
    out = []
    for i in range(Z.shape[0]):
        row_contrib = contrib[i]
        order = np.argsort(-np.abs(row_contrib))[:top_n]
        factors = []
        for j in order:
            feat = coef_order[j]
            factors.append({
                "feature": FEATURE_LABELS.get(feat, feat),
                "detail": magnitude_label(Z[i, j], feat),
                "effect": "positive" if row_contrib[j] > 0 else "negative",
            })
        out.append(factors)
    return out

# --- Option C: gradient boosted trees (non-linear baseline) ---
gbc = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=2, learning_rate=0.05).fit(Xc_tr_s, y_tr)
proba_gbc = gbc.predict_proba(Xc_te_s)[:, 1]
results["gradient_boosting"] = {
    "auc": roc_auc_score(y_te, proba_gbc),
    "brier": brier_score_loss(y_te, proba_gbc),
}

# --- Option D: deep learning MLP ---
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

Xc_tr_t = torch.tensor(Xc_tr_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
Xc_te_t = torch.tensor(Xc_te_s, dtype=torch.float32)

model = MLP(Xc_tr_t.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()

best_auc, best_state = 0, None
for epoch in range(300):
    model.train()
    opt.zero_grad()
    out = model(Xc_tr_t)
    loss = lossf(out, y_tr_t)
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_proba = torch.sigmoid(model(Xc_te_t)).numpy()
        auc = roc_auc_score(y_te, val_proba)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    proba_mlp = torch.sigmoid(model(Xc_te_t)).numpy()
results["deep_learning_mlp"] = {
    "auc": roc_auc_score(y_te, proba_mlp),
    "brier": brier_score_loss(y_te, proba_mlp),
}

print(json.dumps(results, indent=2))
print("\nTop logistic regression coefficients (standardized):")
print(coef.head(10))

# --- Save artifacts for the report/site ---
roc_data = {}
for name, proba in [("draft_position_only", proba_pick), ("logistic_regression", proba_lr),
                     ("gradient_boosting", proba_gbc), ("deep_learning_mlp", proba_mlp)]:
    fpr, tpr, _ = roc_curve(y_te, proba)
    roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

with open(os.path.join(OUT_DIR, "model_results.json"), "w") as f:
    json.dump({"metrics": results, "roc": roc_data, "top_features": coef.head(10).to_dict()}, f)

# --- Score the recent prospects (2019-2023 combine) for the demo ---
Xp_demo = prep_X(prospects, FEATURES + DRAFT_FEATURE)
Xp_demo_i = imp_c.transform(Xp_demo)
Xp_demo_s = scaler_c.transform(Xp_demo_i)
with torch.no_grad():
    prospects_proba_mlp = torch.sigmoid(model(torch.tensor(Xp_demo_s, dtype=torch.float32))).numpy()
prospects_proba_lr = lr.predict_proba(Xp_demo_s)[:, 1]
prospects_proba_gbc = gbc.predict_proba(Xp_demo_s)[:, 1]
prospects_out = prospects[["player_name", "season", "position", "overall_pick", "games_played"]].copy()
prospects_out["predicted_rotation_prob_mlp"] = prospects_proba_mlp
prospects_out["predicted_rotation_prob_lr"] = prospects_proba_lr
prospects_out["combine_grade"] = np.round(prospects_proba_gbc * 100, 1)
prospects_out["factors"] = [json.dumps(f) for f in explain_rows(Xp_demo_s)]
prospects_out.to_csv(os.path.join(OUT_DIR, "prospect_predictions.csv"), index=False)

# --- Also score the held-out test set (known outcomes) for a "model vs reality" table ---
test_out = train_universe.loc[idx_te, ["player_name", "season", "position", "overall_pick", "games_played", "rotation_or_better"]].copy()
test_out["predicted_prob_mlp"] = proba_mlp
test_out["predicted_prob_lr"] = proba_lr
test_out["predicted_prob_pickonly"] = proba_pick
test_out["combine_grade"] = np.round(proba_gbc * 100, 1)
test_out["factors"] = [json.dumps(f) for f in explain_rows(Xc_te_s)]
test_out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

print("\nSaved model_results.json, prospect_predictions.csv, test_predictions.csv")
