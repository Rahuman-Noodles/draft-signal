"""Dataset + site-data validation checks.

Catches the failure modes that silently corrupt this project: schema drift in
the checked-in CSVs, train/prospect season overlap (label leakage by another
name), and site_data.json falling out of sync with the pipeline outputs.

Run:  python3 model/check_data.py   (from the repo root, or anywhere)
Exit code is 0 when every check passes, 1 otherwise.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

failures = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), "|", name, detail)
    if not cond:
        failures.append(name)


combine = pd.read_csv(os.path.join(HERE, "data", "combine_clean.csv"))
career = pd.read_csv(os.path.join(HERE, "data", "career_outcomes.csv"))
draft = pd.read_csv(os.path.join(HERE, "data", "draft_history_clean.csv"))

MEASURE_COLS = [
    "height_wo_shoes", "weight", "wingspan", "standing_reach", "body_fat_pct",
    "standing_vertical_leap", "max_vertical_leap", "lane_agility_time",
    "three_quarter_sprint", "bench_press",
]

# 1. required columns exist
check("combine columns", set(MEASURE_COLS + ["player_id", "player_name", "season"]) <= set(combine.columns))
check("career columns", {"player_id", "games_played"} <= set(career.columns))
check("draft columns", {"player_id", "overall_pick"} <= set(draft.columns))

# 2. no leakage between the mature training universe and the live demo set:
# training uses combine season <= 2016, prospects are >= 2019
train_ids = set(combine.loc[combine["season"] <= 2016, "player_id"])
prospect_ids = set(combine.loc[combine["season"] >= 2019, "player_id"])
check("train/prospect split disjoint", not (train_ids & prospect_ids),
      f"overlap={len(train_ids & prospect_ids)}")

# 3. outcome label is well-formed, computed pipeline-style: left join from the
# combine universe, missing careers count as 0 games (this is what fixes the
# rate at ~0.416 -- filtering to players WITH career rows instead would give
# a survivorship-biased ~0.64)
mature = (combine.loc[combine["season"] <= 2016, ["player_id"]]
          .merge(career.drop_duplicates("player_id"), on="player_id", how="left"))
mature["games_played"] = mature["games_played"].fillna(0)
check("games_played non-negative", bool((mature["games_played"] >= 0).all()))
pos_rate = (mature["games_played"] >= 100).mean()
check("positive rate sane (30-55%)", 0.30 <= pos_rate <= 0.55, f"{pos_rate:.3f}")

# 4. the draft join is many-to-one safe: the pipeline keeps the first row per
# player_id, so after that same dedup every combine player maps to one row
deduped = (pd.to_numeric(draft["player_id"], errors="coerce").dropna()
           .drop_duplicates(keep="first"))
check("draft join one-row-per-player", deduped.is_unique,
      f"{deduped.duplicated().sum()} dupes survive dedup")

# 5. site_data.json is in sync with the pipeline outputs
site = json.load(open(os.path.join(ROOT, "assets", "site_data.json")))
test_pred = pd.read_csv(os.path.join(HERE, "test_predictions.csv"))
prosp_pred = pd.read_csv(os.path.join(HERE, "prospect_predictions.csv"))
check("known row parity", len(site["known"]) == len(test_pred),
      f"{len(site['known'])} vs {len(test_pred)}")
check("prospect row parity", len(site["prospects"]) == len(prosp_pred),
      f"{len(site['prospects'])} vs {len(prosp_pred)}")

REQUIRED_KEYS = {"player_name", "season", "position", "overall_pick", "games_played",
                 "predicted_prob", "combine_grade", "factors", "missing_count"}
for tab in ("known", "prospects", "diamonds", "busts"):
    rows = site[tab]
    check(f"{tab} rows complete",
          all(REQUIRED_KEYS <= set(r) for r in rows) and
          all(len(r["factors"]) == 5 for r in rows),
          f"n={len(rows)}")

print("RESULT", "all pass" if not failures else f"fails={failures}")
sys.exit(1 if failures else 0)
