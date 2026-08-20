# Draft Signal — NBA Draft Evaluation, Deep Learning + Product Case Study

**Live demo:** [Draft Signal](https://www.perplexity.ai/computer/a/draft-signal-nba-draft-evaluat-MORVquEUR8Gmi4nC80k2zQ)

Does NBA combine testing (height, wingspan, vertical leap, sprint/agility times, bench press...) actually predict
which drafted players become rotation-caliber NBA players — or is draft position all that matters? This project
answers that question with real data (24 years of combine results, draft history, and 13.6 million play-by-play
events), compares four modeling approaches (including a PyTorch deep learning model), and packages the result as a
**product management case study**: problem statement, user segment, solution options, an MVP recommendation, and a
metric tree — the same framing you'd use to pitch a feature to an NBA analytics org.

It was built as a portfolio piece for product management / data science interview prep, showing an end-to-end
project: messy real-world data → feature engineering → model comparison → a shippable product decision.

## What's in this repo

```
.
├── index.html, styles.css, app.js   # the web app (static site, no backend needed)
├── assets/                          # charts + precomputed data the site reads
│   ├── auc_comparison.png
│   ├── roc_comparison.png
│   ├── feature_importance.png
│   └── site_data.json               # player-level predictions consumed by app.js
├── PRD.md                           # full PRD-lite write-up (problem, users, options, MVP, metrics)
└── model/                           # everything needed to reproduce the model from scratch
    ├── data/                        # small, pre-extracted CSVs (checked in, no huge downloads needed)
    │   ├── combine_clean.csv
    │   ├── career_outcomes.csv
    │   └── draft_history_clean.csv
    ├── build_dataset.py             # (optional) regenerates the CSVs above from the raw Kaggle DB
    ├── model_pipeline.py            # trains all 4 models, saves metrics + predictions
    ├── make_charts.py               # renders the 3 chart PNGs used on the site
    ├── build_site_data.py           # turns model predictions into assets/site_data.json
    ├── model_results.json           # output of model_pipeline.py (metrics, ROC curves, coefficients)
    └── requirements.txt
```

## The web app

A single static page (no backend, no build step) with three parts:

1. **Model results** — an AUC comparison bar chart, ROC curves for all four models, and a feature-importance chart
   showing which combine measurements matter most.
2. **Player explorer** — a searchable, sortable table with four views:
   - *Known outcomes*: real players drafted 2000–2016, model prediction vs. what actually happened
   - *Current prospects*: recent draftees (2019–2023) scored live by the model, career still unfolding
   - *Diamonds in the rough*: players the model scored low who became rotation players anyway
   - *Overrated on paper*: players the model scored high who never stuck in the league
3. **Combine Grade Report Card (the working MVP)** — click any player row to open a report card with a 0–100
   combine grade (the gradient-boosted model's win probability, rescaled) and the top 5 factors driving that score
   in plain language (e.g. "Standing reach: above average", "Lane agility time: faster than typical"), each tagged
   as a positive or negative contributor. This is the PRD's MVP recommendation, actually shipped and clickable
   rather than just described — see `model/model_pipeline.py`'s `explain_rows()` for how the explanations are
   computed (standardized logistic-regression coefficients × each player's own standardized feature values).
4. **PRD-lite** — the full product write-up, embedded directly in the page.

### Running the site locally

No build tools needed — it's plain HTML/CSS/JS.

```bash
cd /path/to/draft-signal
python3 -m http.server 8000
# then open http://localhost:8000
```

## The model

**Question:** given only information available at draft time (combine measurements + draft position), can we
predict whether a player becomes at least a rotation player (100+ career NBA games)?

**Data source:** [NBA Database, wyattowalsh (Kaggle)](https://www.kaggle.com/datasets/wyattowalsh/basketball) — a
2.2GB SQLite database covering 64,698 games, 1,633 combine records, and 13.6 million play-by-play events, sourced
from stats.nba.com.

**Why this wasn't a simple `pd.read_csv` job:** the dataset has no ready-made "career length" field, and its
`common_player_info` table is missing well-known long-career players entirely (e.g. Jamal Crawford's 19-year career
is absent there). Career outcomes were instead reconstructed from the 13.6M-row play-by-play log — counting
distinct games where a player appears as a shooter, passer, rebounder, or substitute — which is the approach used
in `model/build_dataset.py`.

**Models compared** (trained on 1,130 players from combine classes 2000–2016, evaluated on a held-out 25% test split):

| Model | AUC | Brier score |
|---|---|---|
| Draft position only (status quo) | 0.686 | 0.208 |
| Logistic regression (+ combine data) | 0.735 | 0.196 |
| Gradient boosted trees | **0.842** | **0.154** |
| Deep learning (PyTorch MLP) | 0.795 | 0.166 |

**Takeaway:** combine data adds real, meaningful predictive signal beyond draft slot. Gradient boosting wins
overall; the neural network beats the linear baseline but loses to boosting — a realistic outcome given the
training set is only ~1,130 rows, too small for a neural net's extra capacity to pay off over a well-tuned tree
ensemble. See `PRD.md` for how this shapes the MVP recommendation.

### Reproducing the model

```bash
cd model
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~200MB

python3 model_pipeline.py     # trains all 4 models -> model_results.json, *_predictions.csv
python3 make_charts.py        # renders the 3 PNGs into ../assets/
python3 build_site_data.py    # rebuilds ../assets/site_data.json for the web app
```

All three scripts run entirely off the small CSVs already checked into `model/data/` — no multi-gigabyte download
required. If you want to regenerate `model/data/*.csv` from scratch (e.g. against a newer version of the dataset),
download `nba.sqlite` from the [Kaggle dataset page](https://www.kaggle.com/datasets/wyattowalsh/basketball), place
it at `model/raw/nba.sqlite`, and run `python3 build_dataset.py` first.

## The product case study

The full PRD-lite is in [`PRD.md`](./PRD.md) and covers:

- **Problem:** scouting staff have no systematic way to weigh 40+ combine metrics against actual career outcomes
- **User segment:** NBA front-office / analytics staff building draft boards, plus player-development staff
- **4 solution options:** rule-based scorecard → interpretable logistic regression → gradient boosting → deep
  learning, each with an explicit tradeoff (interpretability vs. accuracy vs. data requirements)
- **MVP cut:** ship the gradient-boosted model as a 0–100 "combine grade" alongside existing scouting reports, with
  low-confidence flags for prospects who skip combine testing (a real, growing problem — several recent top picks
  opt out of athletic testing entirely)
- **Metric tree:** north star (draft-pick surplus value), model health (AUC/Brier per draft class), adoption
  (% of scouting reports citing the grade), and a trust guardrail (rate of high-confidence predictions overturned
  by scouts)

## Known limitations

- Career outcome labels are a play-by-play-derived proxy, not an official box-score count.
- Training data covers combine classes 2000–2016 only, so outcomes have had time to mature; 2019–2023 prospects
  shown in the demo are live, unvalidated predictions.
- Several sparse "shooting drill" combine measurements (e.g. spot-up shooting by court zone) were excluded from
  the feature set — under 15% of players have non-missing values for these, which made imputation unstable across
  train/test splits.
- The training set (~1,130 labeled rows) is the binding constraint on how much a deep learning approach can
  outperform gradient boosting — a data-collection recommendation as much as a modeling one.

## Data & attribution

Data: [NBA Database, wyattowalsh (Kaggle)](https://www.kaggle.com/datasets/wyattowalsh/basketball), originally
sourced from stats.nba.com. This project is an independent analysis and is not affiliated with or endorsed by the
NBA.
