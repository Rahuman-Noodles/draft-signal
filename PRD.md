# Draft Signal — PRD-lite
### Predicting NBA draft-pick career outcomes from combine data (Product Management case study)

## Background

NBA teams run every draft-eligible prospect through the annual scouting combine — height/weight, wingspan,
standing reach, vertical leap, sprint and agility drills, body fat, bench press — but there is no standardized way
to translate those numbers into a probability-weighted career forecast. In practice, most evaluation reverts to
draft position as the implicit forecast, even though draft slot is a subjective human judgment made before the
player has logged a single NBA possession.

This project uses the public [NBA Database (Kaggle, wyattowalsh)](https://www.kaggle.com/datasets/wyattowalsh/basketball) —
64,698 games, 1,633 combine records, and 13.6 million play-by-play events dating back to the 1940s, sourced from
stats.nba.com — to test whether combine measurements meaningfully predict which drafted players become
rotation-caliber NBA players, and to scope a product around the answer.

## Problem

Scouting departments have no systematic, validated way to weigh 40+ combine metrics against actual career outcomes.
Evaluation is qualitative and inconsistent scout-to-scout, and there's no measurable way to justify continued
investment in combine testing versus relying purely on film and pre-draft rankings.

## User segment

- **Primary:** NBA front-office / basketball analytics staff building draft boards ahead of the draft.
- **Secondary:** Player-development staff who want an early read on a rookie's career-trajectory risk to plan
  coaching and development resource allocation in year one.

## Data & methodology (what grounds this PRD)

- **Outcome label:** `rotation_or_better` = played 100+ career NBA games. The dataset has no ready-made
  career-length field, so this was reconstructed from 13.6M play-by-play rows (every event where a player appears
  as shooter, passer, rebounder, or substitute), which also caught real data-quality issues — e.g., the database's
  own `common_player_info` table is missing well-known long-career players (Jamal Crawford's 19-year career is
  absent there), so it could not be used as ground truth on its own.
- **Predictors:** combine measurements available at draft time — height, weight, wingspan, standing reach, body
  fat %, standing/max vertical leap, lane agility time, three-quarter sprint, bench press — plus overall draft pick.
- **Training set:** 1,130 players from combine classes 2000–2016 (old enough that career outcomes have matured),
  41.6% positive rate.
- **Held-out test set:** drawn from the same 2000–2016 pool, outcomes known.
- **Live demo set:** 341 players from combine classes 2019–2023, whose careers are still unfolding — used to show
  what the model says about current, unproven prospects.
- **Four models compared** on the held-out test set:

| Model | AUC | Brier score | What it represents |
|---|---|---|---|
| Draft position only | 0.686 | 0.208 | Status quo — what teams effectively use today |
| Logistic regression (+ combine) | 0.735 | 0.196 | Interpretable baseline |
| Gradient boosted trees | **0.842** | **0.154** | Best accuracy, captures non-linear interactions |
| Deep learning (MLP) | 0.795 | 0.166 | Highest ceiling long-term, but underperforms boosting on ~1,100 rows |

**Headline finding:** combine testing carries real, measurable signal beyond draft slot — a 0.686 → 0.842 AUC
lift is a large, decision-relevant improvement, not noise. The deep learning model beats the linear baseline but
loses to gradient boosting today; with this data volume, tree ensembles are the better bet, and a neural
architecture is worth revisiting only once more seasons of tracking data are folded in.

**Top standardized predictors (logistic regression):** overall pick (strongest, negative — earlier pick = better
odds), lane agility time (negative — faster is better), standing reach (positive), body fat % (negative), bench
press (negative, direction worth further investigation), standing/max vertical leap (positive), three-quarter
sprint (positive).

## Solution options considered

1. **Rule-based combine scorecard** — simple weighted thresholds on 2–3 top metrics (e.g., lane agility + standing
   reach). Fastest to ship, fully transparent to scouts, but leaves most of the 0.69→0.84 AUC gap on the table.
2. **Interpretable baseline (logistic regression)** — every coefficient explainable in one sentence to a
   non-technical scout; 0.735 AUC, a real but modest lift over the status quo.
3. **Gradient-boosted ensemble** — best accuracy (0.842 AUC), captures interaction effects (e.g., height only
   matters in combination with a certain wingspan), at some cost to interpretability versus option 2.
4. **Deep learning trajectory model** — highest long-term potential if extended with season-by-season performance
   sequences rather than a single combine snapshot, but currently underperforms gradient boosting because the
   labeled dataset (~1,130 rows) is too small for the extra model capacity to pay off.

## MVP cut

Ship the **gradient-boosted model** behind a simple 0–100 "combine grade," shown alongside the existing scouting
report, with the top 5 contributing factors surfaced in plain language per prospect. Hold the deep learning
approach as a v2 bet, gated on collecting more longitudinal tracking data (multi-season measurements, not just a
single pre-draft snapshot).

**Shipped:** this is implemented and clickable in the live demo — select any player row in the Player Explorer to
open its Combine Grade Report Card (grade + top-5 factors).

**Explicitly flag low-confidence predictions** for prospects who skip parts of the combine — a real and growing
failure mode, since several of the most hyped recent prospects (including consensus top picks) now opt out of
athletic testing entirely, which forces the model back toward pick order alone for exactly the players evaluators
most want a second opinion on. Solving this gap (e.g., backfilling with predraft workout data, agility footage, or
college performance tracking) is the top follow-on bet after MVP launch.

## Metric tree

- **North star:** Draft-pick surplus value — career win-shares delivered relative to draft-slot expectation for
  picks the tool scored confidently.
- **Model health (leading indicator):** AUC and Brier score recalculated on each newly matured draft class,
  refreshed annually as outcomes resolve.
- **Adoption (engagement):** % of scouting reports where staff cite the combine grade in their final
  recommendation; time-to-decision on draft-board ranking sessions.
- **Trust / guardrail:** rate of high-confidence model predictions overturned by scouts post-hoc — a rising rate
  signals model drift or a blind spot (e.g., a new athletic profile the model hasn't seen) requiring retraining.

## Known limitations

- Career outcome labels are reconstructed from play-by-play appearances, a reasonable proxy but not an official
  box-score-derived count.
- Training restricted to 2000–2016 classes so careers have had time to mature; 2019–2023 prospects shown in the
  demo are live, unvalidated predictions, not confirmed outcomes.
- Sparse "shooting drill" combine measurements (spot-up shooting percentages by court zone) were excluded from the
  final feature set — fewer than 15% of players have non-missing values for these, which made per-split imputation
  unstable; a production version would need a dedicated missing-data strategy to use them.
- Sample size (~1,130 labeled training rows) is the binding constraint on how much a deep learning approach can
  outperform gradient boosting; this is itself a data-collection recommendation, not just a modeling limitation.

## Source

Data: [NBA Database, wyattowalsh (Kaggle)](https://www.kaggle.com/datasets/wyattowalsh/basketball), sourced from stats.nba.com.
