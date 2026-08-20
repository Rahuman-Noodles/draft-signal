"""
One-time script to regenerate model/data/career_outcomes.csv and
model/data/combine_clean.csv from scratch.

NOT required to run the model — those two CSVs are already checked into
model/data/. Only run this if you want to rebuild them yourself, e.g. after
downloading a newer version of the dataset.

Download nba.sqlite from https://www.kaggle.com/datasets/wyattowalsh/basketball
(about 2.2GB) and place it at model/raw/nba.sqlite before running this script.
"""
import os, sqlite3, pandas as pd, numpy as np, time

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "raw", "nba.sqlite")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)
con = sqlite3.connect(DB)
cur = con.cursor()

t0 = time.time()
print("Creating indices on play_by_play (one-time)...")
for col in ["player1_id", "player2_id", "player3_id"]:
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pbp_{col} ON play_by_play({col})")
con.commit()
print(f"Indices done in {time.time()-t0:.1f}s")

# Combine stats: the athletic testing feature table
dc = pd.read_sql("SELECT * FROM draft_combine_stats", con)
dc = dc.drop_duplicates(subset=["player_id"], keep="last")  # keep most recent combine entry per player
player_ids = dc["player_id"].dropna().astype(int).unique().tolist()
print("unique combine players:", len(player_ids))

# temp table of player ids for fast joins
cur.execute("DROP TABLE IF EXISTS combine_ids")
cur.execute("CREATE TEMP TABLE combine_ids (player_id INTEGER PRIMARY KEY)")
cur.executemany("INSERT INTO combine_ids VALUES (?)", [(int(p),) for p in player_ids])
con.commit()

t0 = time.time()
q = """
WITH appearances AS (
    SELECT player1_id AS pid, game_id FROM play_by_play WHERE player1_id IN (SELECT player_id FROM combine_ids)
    UNION
    SELECT player2_id AS pid, game_id FROM play_by_play WHERE player2_id IN (SELECT player_id FROM combine_ids)
    UNION
    SELECT player3_id AS pid, game_id FROM play_by_play WHERE player3_id IN (SELECT player_id FROM combine_ids)
)
SELECT a.pid AS player_id,
       COUNT(DISTINCT a.game_id) AS games_played,
       MIN(g.game_date) AS first_game_date,
       MAX(g.game_date) AS last_game_date
FROM appearances a
JOIN game g ON g.game_id = a.game_id
GROUP BY a.pid
"""
career = pd.read_sql(q, con)
print(f"career aggregation done in {time.time()-t0:.1f}s, rows: {len(career)}")
career.to_csv(os.path.join(DATA_DIR, "career_outcomes.csv"), index=False)
dc.to_csv(os.path.join(DATA_DIR, "combine_clean.csv"), index=False)
print("saved career_outcomes.csv and combine_clean.csv")
