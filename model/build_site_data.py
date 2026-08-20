"""
Builds assets/site_data.json for the web app from model_pipeline.py's output
(test_predictions.csv, prospect_predictions.csv). Run model_pipeline.py first.
"""
import os
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(HERE), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

t = pd.read_csv(os.path.join(HERE, "test_predictions.csv"))
p = pd.read_csv(os.path.join(HERE, "prospect_predictions.csv"))


def clean(df, prob_col):
    df = df.copy()
    df["overall_pick"] = pd.to_numeric(df["overall_pick"], errors="coerce").fillna(-1).astype(int)
    df["games_played"] = pd.to_numeric(df["games_played"], errors="coerce").fillna(0).astype(int)
    df[prob_col] = (df[prob_col] * 100).round(1)
    df["position"] = df["position"].fillna("\u2014")
    return df


t = clean(t, "predicted_prob_mlp")
p = clean(p, "predicted_rotation_prob_mlp")

known = (
    t[["player_name", "season", "position", "overall_pick", "games_played", "rotation_or_better", "predicted_prob_mlp"]]
    .rename(columns={"predicted_prob_mlp": "predicted_prob", "rotation_or_better": "actual_hit"})
    .sort_values("season", ascending=False)
    .to_dict(orient="records")
)

prospects = (
    p[["player_name", "season", "position", "overall_pick", "games_played", "predicted_rotation_prob_mlp"]]
    .rename(columns={"predicted_rotation_prob_mlp": "predicted_prob"})
    .sort_values("predicted_prob", ascending=False)
    .to_dict(orient="records")
)

# "diamonds in the rough": model scored them low, but they became rotation-or-better
diamonds = [r for r in known if r["predicted_prob"] < 45 and r["actual_hit"] == 1]
diamonds = sorted(diamonds, key=lambda r: r["predicted_prob"])[:8]

# "overrated on paper": model scored them high, but they never became rotation-or-better
busts = [r for r in known if r["predicted_prob"] > 75 and r["actual_hit"] == 0]
busts = sorted(busts, key=lambda r: -r["predicted_prob"])[:8]

out = {"known": known, "prospects": prospects, "diamonds": diamonds, "busts": busts}

with open(os.path.join(ASSETS_DIR, "site_data.json"), "w") as f:
    json.dump(out, f, allow_nan=False)

print(f"known={len(known)} prospects={len(prospects)} diamonds={len(diamonds)} busts={len(busts)}")
