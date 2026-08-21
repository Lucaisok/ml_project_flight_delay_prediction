# Flight Delay Prediction

A machine learning project to predict flight delay duration (in minutes) using Tunisair operational scheduling data.

## Problem

Flight delays are costly for airlines beyond passenger frustration: they reduce operational efficiency, increase capital costs from aircraft repositioning, and force expensive crew reallocation when duty-time limits are hit. Severe, memorable delays also erode passenger trust and future demand.

**Goal:** predict the delay duration (in minutes) for each flight, framed as a regression problem, to enable proactive operational planning instead of reactive firefighting.

## Data

- **107,833** flight records, of which **64.6%** experienced some delay.
- Target (`target`, delay in minutes): mean **48.7**, median **14**, std **117.1**, range **0–3,451**.
- The distribution is strongly right-skewed: 25% of flights have zero delay, but a small number of extreme outliers pull the mean well above the median.
- **Key finding:** severely delayed flights (180+ minutes) make up just **6.5%** of the data but account for **86.6%** of total squared deviation — directly motivating the choice of evaluation metric below.

Raw columns: `ID`, `DATOP`, `FLTID`, `DEPSTN`, `ARRSTN`, `STD`, `STA`, `STATUS`, `AC`, `target`.

## Metric

**RMSE (primary)**, **MAE (secondary)**.

Delay costs scale non-linearly — a short delay is absorbed by schedule buffer, while a long delay triggers crew reallocation, aircraft repositioning, and lost passenger trust. RMSE penalizes large errors more heavily, aligning the evaluation with where the real business cost concentrates. MAE is reported alongside to track everyday operational accuracy.

## Methodology

1. **Data cleaning** — parsed inconsistent datetime formats (`STA`/`STD` mixed `HH:MM:SS` / `HH.MM.SS`), split composite identifiers (`FLTID` → airline + flight number, `AC` → airline + aircraft type + tail number), and **removed the `STATUS` column** after confirming it was a target-leakage risk (rows with `SCH`/`DEL` status had `target = 0` by construction).

2. **Feature engineering** (15 features), including:
   - `route`, `flight_duration`, `crosses_year`, `is_weekend`, `is_night`, `is_holiday`, `flight_season`
   - Same-day delay propagation per aircraft / airline / route / departure station
   - Delay of each aircraft's immediately preceding flight (regardless of day)
   - Historical average delay per route / airline / departure & arrival station
   - A 3-flight rolling average of prior delay per aircraft

   Target-derived features (propagation, historical averages) are computed as strictly backward-looking, time-ordered aggregates — no row ever uses its own or a future row's target value.

3. **Time-based train/test split** — trained on earlier flights, tested on the most recent ~20% of dates. A random split was ruled out because several features are time-dependent; a random split could let future flights leak into a "past" flight's history.

4. **Baselines** — constant (mean/median), route-mean, and Ridge regression, to establish the floor any real model needs to beat.

5. **Advanced models** — Random Forest and XGBoost, each hyperparameter-tuned via `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation.

## Results

| Model | RMSE | MAE |
|---|---|---|
| Constant (median) | 150.14 | 60.48 |
| Route mean | 139.60 | 61.87 |
| Ridge regression | 139.47 | 60.67 |
| Random Forest (tuned) | 121.57 | 56.31 |
| XGBoost (tuned) | 122.02 | 54.13 |
| **XGBoost + Tweedie objective + rolling delay feature** | **120.98** | **49.40** |

**Feature importance:** same-day and prior-flight delay propagation for a specific aircraft are by far the strongest predictors — stronger than calendar effects (season, holidays, weekday). Delays are driven far more by "what's happening right now" than by "what usually happens on this route or time of year."

**Error analysis:** the model is accurate on typical flights (RMSE ~55–70 min across Early/Minor/Moderate/Major severity buckets) but error rises sharply on the rare Severe (180+ min) segment (RMSE ~350, MAE ~222) — the intended focus of the RMSE-first metric choice, and the clearest target for further improvement.

## Limitations & Future Work

- The model still under-predicts the most extreme delays (1,000+ minutes) — the rarest, hardest cases to learn from.

## Presentation

See `presentation/flight_delay_presentation.pdf` for the full findings, recommendation, and proposed data product (a delay-risk dashboard for operations), built for a 10-minute non-technical stakeholder audience.
