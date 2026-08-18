import json

def md(text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md("""# Flight Delay Prediction: Baseline Model

Working notebook for the [Zindi Flight Delay Prediction Challenge](https://zindi.africa/competitions/flight-delay-prediction-challenge) (Tunisair). This notebook builds Milestone 1: the **baseline model**, following the plan in `Flight_Delay_Baseline_and_Strategy_Report.pdf`.

## Define the Business Goal

- **Stakeholder:** Tunisair operations / the Zindi organisers evaluating submissions.
- **Prediction:** the delay of a given Tunisair flight, **in minutes** (can be negative for an early departure/arrival).
- **Problem type:** regression -- the target is a continuous number, not a delayed/not-delayed label.
- **Evaluation metric: Root Mean Squared Error (RMSE).** RMSE squares each error before averaging, so a few very large misses hurt the score far more than many small ones. This favours models that avoid big outlier misses over models that are only accurate on average, and it is also the metric the Zindi leaderboard scores submissions on.
- **Baseline model:** the simplest rule that makes a prediction. Two are built below -- a **naive mean/median predictor** (the absolute floor) and a **Ridge regression** on basic engineered features (the first "real" ML baseline). Every later, more complex model (Random Forest, LightGBM, CatBoost, ...) must beat these to justify its added complexity."""))

cells.append(md("""## Get the Data

> [!IMPORTANT]
> This repo does not ship the competition data. Download `train.csv`, `test.csv`, and `SampleSubmission.csv` from the **Data** tab of the [challenge page](https://zindi.africa/competitions/flight-delay-prediction-challenge) (Zindi account required) and place them in `data/`.

Expected schema (per the strategy report -- verify against `train.info()` below, column names may differ slightly):

| Column | Meaning |
|---|---|
| `ID` | unique row identifier |
| `DATOP` | date of operation of the flight |
| `FLTID` | flight number (high-cardinality categorical) |
| `DEPSTN` | departure station / airport code |
| `ARRSTN` | arrival station / airport code |
| `STD` | scheduled time of departure |
| `STA` | scheduled time of arrival |
| `STATUS` | flight status code |
| `AC` | aircraft registration/type |
| `target` | delay in minutes (train only) -- what we predict for test |"""))

cells.append(code("""import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["DATOP", "STD", "STA"])
test = pd.read_csv("data/test.csv", parse_dates=["DATOP", "STD", "STA"])
train.head()"""))

cells.append(md("""### First Look at the Data

A quick structural check: column types, missingness, and the target distribution."""))

cells.append(code("""train.info()"""))

cells.append(code("""# The target is right-skewed with a long tail of large delays (and can be negative
# for early flights) -- this shapes both the baseline choice and the evaluation metric's
# behaviour (RMSE punishes the tail hardest).
train["target"].describe()"""))

cells.append(md("""## Train / Validation Split

We use a **time-based split**, not a random train/test split: earlier dates for training, the most recent ~20% of dates held out for validation. This matters for two reasons:

1. Several natural features for this problem (historical average delay per route/aircraft/flight number, delay-propagation from the same aircraft's earlier flight that day) are computed **over time**. A random split would let information from "future" rows leak into a "past" row's historical average, and a random shuffled CV on time-ordered data overstates validation performance.
2. It mirrors how the model will actually be used: predicting delays for flights that have not happened yet, from patterns learned on flights that already have.

A larger iteration of this project would use forward-chaining CV (several expanding time windows) instead of a single cut -- see the strategy report, Section 8."""))

cells.append(code("""train = train.sort_values("DATOP")

# Hold out the most recent ~20% of dates for validation
cutoff = train["DATOP"].quantile(0.8)
X_train, X_val = train[train["DATOP"] <= cutoff], train[train["DATOP"] > cutoff]

print(f"Train: {len(X_train)} rows up to {X_train['DATOP'].max().date()}")
print(f"Validation: {len(X_val)} rows from {X_val['DATOP'].min().date()} onward")"""))

cells.append(md("""## Baseline Models

Before training a real model, we set the RMSE floor with three baselines of increasing sophistication. Each is fit on `X_train` only and scored on `X_val` with RMSE."""))

cells.append(code("""from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))"""))

cells.append(md("""### Baseline 0 -- Constant Predictor

Ignore every feature; always predict the training set's mean delay. If a later model cannot beat this, it has learned nothing useful."""))

cells.append(code("""baseline0_pred = np.full(len(X_val), X_train["target"].mean())
rmse_baseline0 = rmse(X_val["target"], baseline0_pred)

print(f"Baseline 0 (constant mean) always predicts: {X_train['target'].mean():.2f} minutes")
print(f"Baseline 0 validation RMSE: {rmse_baseline0:.2f}")"""))

cells.append(md("""### Baseline 1 -- Group-Mean Predictor

Predict the historical mean delay of the flight's **route** (`DEPSTN`-`ARRSTN`), computed on training data only. No modelling required, but it already captures "this route is chronically late" signal."""))

cells.append(code("""route_mean = (
    X_train.assign(route=X_train["DEPSTN"].astype(str) + "_" + X_train["ARRSTN"].astype(str))
    .groupby("route")["target"].mean()
)
global_mean = X_train["target"].mean()  # fallback for routes not seen in training

val_route = X_val["DEPSTN"].astype(str) + "_" + X_val["ARRSTN"].astype(str)
baseline1_pred = val_route.map(route_mean).fillna(global_mean)
rmse_baseline1 = rmse(X_val["target"], baseline1_pred)

print(f"Baseline 1 (route mean) validation RMSE: {rmse_baseline1:.2f}")"""))

cells.append(md("""### Baseline 2 -- Ridge Regression

A simple linear model over basic engineered features: hour of day, day of week, month, scheduled block time, and route. This is the first baseline that is a real (if simple) ML model, and the reference point for whether tree ensembles (Random Forest, LightGBM, CatBoost -- the next iteration) are actually earning their extra complexity."""))

cells.append(code("""from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def add_features(df):
    df = df.copy()
    df["hour"] = df["STD"].dt.hour
    df["dow"] = df["STD"].dt.dayofweek
    df["month"] = df["STD"].dt.month
    df["block_min"] = (df["STA"] - df["STD"]).dt.total_seconds() / 60
    df["route"] = df["DEPSTN"].astype(str) + "_" + df["ARRSTN"].astype(str)
    return df

X_train_feat = add_features(X_train)
X_val_feat = add_features(X_val)

cat_cols = ["route", "DEPSTN", "ARRSTN"]
num_cols = ["hour", "dow", "month", "block_min"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

baseline2 = Pipeline([("preprocess", preprocess), ("reg", Ridge(alpha=1.0))])
baseline2.fit(X_train_feat[cat_cols + num_cols], X_train_feat["target"])

baseline2_pred = baseline2.predict(X_val_feat[cat_cols + num_cols])
rmse_baseline2 = rmse(X_val["target"], baseline2_pred)

print(f"Baseline 2 (Ridge regression) validation RMSE: {rmse_baseline2:.2f}")"""))

cells.append(md("""### Compare the Baselines"""))

cells.append(code("""pd.DataFrame({
    "baseline": ["Constant mean", "Route mean", "Ridge regression"],
    "validation_rmse": [rmse_baseline0, rmse_baseline1, rmse_baseline2],
}).sort_values("validation_rmse")"""))

cells.append(md("""Whichever baseline scores lowest here is the number every subsequent model (Random Forest, then gradient-boosted trees -- LightGBM / CatBoost, per the strategy report's recommendation) has to beat. Record this score before moving on to feature engineering and more advanced models.

## Next Steps

Per the strategy report's roadmap (Sections 3, 6-7):

1. Expand feature engineering: cyclical hour/month encoding, part-of-day buckets, an airport-traffic proxy, and -- carefully, using only past rows -- expanding-mean delay per aircraft/route/flight number and same-day delay-propagation.
2. Move to gradient-boosted trees (CatBoost first, for native handling of `FLTID`/`DEPSTN`/`ARRSTN`/`AC`; LightGBM alongside it for fast iteration), tuned with forward-chaining time-based CV and early stopping.
3. Try `log1p(target)` or a Huber loss to reduce the influence of extreme delays, comparing against raw-minute training on the same validation split.
4. Blend/stack the strongest 2-3 models for the final submission.
5. Do error analysis by route, month, and delay magnitude before the final retrain-on-everything and `submission.csv` write-out (columns `id`, `target`, matching `SampleSubmission.csv` exactly)."""))

cells.append(md("""## References & Further Reading

- [**Flight Delay Prediction Challenge (Zindi)**](https://zindi.africa/competitions/flight-delay-prediction-challenge): the competition this notebook targets.
- `Flight_Delay_Baseline_and_Strategy_Report.pdf`: the fuller strategy report this notebook implements (data caveats, feature ideas, model comparison, validation pitfalls).
- [**root_mean_squared_error (scikit-learn)**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html): the evaluation metric used here and by the leaderboard.
- [**Ridge (scikit-learn)**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html): the model behind Baseline 2.
- [**Common pitfalls and recommended practices (scikit-learn)**](https://scikit-learn.org/stable/common_pitfalls.html): general guidance on data leakage, directly relevant to the historical-average features planned for the next iteration."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("04_flight_delay_modeling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("wrote 04_flight_delay_modeling.ipynb with", len(cells), "cells")
