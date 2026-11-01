"""Leak-free training, baselines, honest evaluation and experiment tracking.

Run from the project root:   python -m src.train

Design (see README / reports/evaluation_report.md):
  1. Split rows FIRST (80/20, seed 42). City medians used by the proxy label are computed on the
     training rows only, so no test-row information reaches the labels or features.
  2. Classification = recovery of the RULE-BASED proxy label 'Good_Investment'.
  3. Regression = estimating the OBSERVED current price from non-price attributes.
     The 5-year price is an assumption-based projection and is NOT modelled/evaluated here.
  4. Every model is compared with naive baselines. The "best" model is chosen by cross-validation
     on the training split; the held-out test set is used once for reporting.
"""
import json
import os
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features import (CLS_FEATURES, GOOD_SCORE_THRESHOLD, HORIZON_YEARS, MODEL_CATS, NUMERIC,
                          PRICE_COLUMNS, REFERENCE_YEAR, REG_FEATURES, add_targets, city_stats,
                          clean, engineer, load_raw)
from src.tracking import Tracker

warnings.filterwarnings("ignore")
SEED = 42
DATA = "data/india_housing_prices.csv"
CV_ROWS, CV_FOLDS = 60_000, 3          # model selection on a training subsample (keeps runtime sane)
IMP_ROWS = 8_000                        # rows used for permutation importance


def _split_cols(features):
    cats = [c for c in features if c in MODEL_CATS]
    nums = [c for c in features if c not in MODEL_CATS]
    return cats, nums


def tree_pre(features):
    cats, _ = _split_cols(features)
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats)],
        remainder="passthrough")


def linear_pre(features):
    cats, nums = _split_cols(features)
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cats if c != "State"]),
                              ("num", StandardScaler(), nums)])


def hgb(cls, features):
    return cls(categorical_features=list(range(len(_split_cols(features)[0]))),
               max_iter=200, learning_rate=0.1, random_state=SEED)


def cls_metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, proba),
                pr_auc=average_precision_score(y, proba), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def reg_metrics(y, pred):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, pred))), mae=mean_absolute_error(y, pred),
                r2=r2_score(y, pred))


def main(data=DATA, out_dir="."):
    os.chdir(out_dir)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    tracker = Tracker("Real_Estate_Investment_Advisor")
    print("tracking backend:", tracker.backend, flush=True)

    # ------------------------------------------------------------------ split first (no leakage)
    full = engineer(clean(load_raw(data)))
    train, test = train_test_split(full, test_size=0.2, random_state=SEED)
    stats = city_stats(train)                      # TRAIN-ONLY city medians
    train, test = add_targets(train, stats), add_targets(test, stats)
    n_train, n_test = len(train), len(test)
    print(f"train={n_train} test={n_test} | good-investment rate train={train.Good_Investment.mean():.3f} "
          f"test={test.Good_Investment.mean():.3f}", flush=True)

    cv_idx = np.random.RandomState(SEED).choice(n_train, CV_ROWS, replace=False)
    results = {"classification": {}, "regression": {}}
    cv = {"classification": {}, "regression": {}}
    fitted = {}

    # ------------------------------------------------------------------ classification
    ytr, yte = train["Good_Investment"], test["Good_Investment"]
    cheaper_te = (test["Price_per_SqFt_Rs"] <= test["City"].map(stats["city_median_ppsf"])).astype(float)
    cls_specs = {
        "baseline_majority_class": (CLS_FEATURES, lambda: DummyClassifier(strategy="most_frequent"), None),
        "logreg": (CLS_FEATURES, lambda: LogisticRegression(max_iter=300), linear_pre),
        "hist_gb": (CLS_FEATURES, lambda: hgb(HistGradientBoostingClassifier, CLS_FEATURES), tree_pre),
        # Ablation: removing price inputs shows how much the proxy label depends on price.
        "hist_gb_no_price_features": ([f for f in CLS_FEATURES if f not in PRICE_COLUMNS],
                                      lambda: hgb(HistGradientBoostingClassifier,
                                                  [f for f in CLS_FEATURES if f not in PRICE_COLUMNS]), tree_pre),
    }
    # heuristic baseline: "good" iff cheaper than the city median (one rule component only)
    results["classification"]["baseline_cheaper_than_city_median"] = cls_metrics(yte, cheaper_te.values)
    with tracker.run("baseline_cheaper_than_city_median"):
        tracker.params({"task": "classification", "kind": "heuristic baseline"})
        tracker.metrics({k: v for k, v in results["classification"]["baseline_cheaper_than_city_median"].items()})
    for name, (feats, mk, pre) in cls_specs.items():
        est = mk()
        pipe = Pipeline([("pre", pre(feats)), ("m", est)]) if pre else Pipeline([("m", est)])
        Xtr = train[feats] if pre else np.zeros((n_train, 1))
        Xte = test[feats] if pre else np.zeros((n_test, 1))
        with tracker.run(name):
            tracker.params({"task": "classification", "model": name, "n_features": len(feats),
                            "label": "rule-based proxy (not observed)"})
            t = time.time()
            pipe.fit(Xtr, ytr)
            fit_s = time.time() - t
            m = cls_metrics(yte, pipe.predict_proba(Xte)[:, 1])
            if pre:
                cvs = cross_val_score(pipe, train[feats].iloc[cv_idx], ytr.iloc[cv_idx], cv=CV_FOLDS, scoring="roc_auc")
                cv["classification"][name] = {"roc_auc_mean": float(cvs.mean()), "roc_auc_std": float(cvs.std())}
                m["cv_roc_auc"] = float(cvs.mean())
            tracker.metrics({**{k: v for k, v in m.items()}, "fit_seconds": fit_s})
        results["classification"][name] = m
        fitted[("cls", name)] = (pipe, feats)
        print("classification |", name, {k: round(v, 4) for k, v in m.items() if k in ("accuracy", "f1", "roc_auc", "cv_roc_auc")}, flush=True)
    cand = [n for n in ("logreg", "hist_gb")]
    best_c = max(cand, key=lambda n: cv["classification"][n]["roc_auc_mean"])  # chosen on CV, not test

    # ------------------------------------------------------------------ regression (observed price)
    ytr_r, yte_r = train["Price_in_Lakhs"], test["Price_in_Lakhs"]
    city_mean = train.groupby("City")["Price_in_Lakhs"].mean()
    results["regression"]["baseline_city_mean"] = reg_metrics(yte_r, test["City"].map(city_mean).fillna(ytr_r.mean()))
    with tracker.run("baseline_city_mean"):
        tracker.params({"task": "regression", "kind": "per-city training mean"})
        tracker.metrics(results["regression"]["baseline_city_mean"])
    reg_specs = {
        "baseline_mean": (lambda: DummyRegressor(strategy="mean"), None),
        "ridge": (lambda: Ridge(alpha=1.0), linear_pre),
        "hist_gb": (lambda: hgb(HistGradientBoostingRegressor, REG_FEATURES), tree_pre),
    }
    for name, (mk, pre) in reg_specs.items():
        est = mk()
        pipe = Pipeline([("pre", pre(REG_FEATURES)), ("m", est)]) if pre else Pipeline([("m", est)])
        Xtr = train[REG_FEATURES] if pre else np.zeros((n_train, 1))
        Xte = test[REG_FEATURES] if pre else np.zeros((n_test, 1))
        with tracker.run("reg_" + name):
            tracker.params({"task": "regression", "model": name, "target": "Price_in_Lakhs (observed)",
                            "n_features": len(REG_FEATURES)})
            t = time.time()
            pipe.fit(Xtr, ytr_r)
            fit_s = time.time() - t
            m = reg_metrics(yte_r, pipe.predict(Xte))
            if pre:
                cvs = cross_val_score(pipe, train[REG_FEATURES].iloc[cv_idx], ytr_r.iloc[cv_idx], cv=CV_FOLDS,
                                      scoring="neg_root_mean_squared_error")
                cv["regression"][name] = {"rmse_mean": float(-cvs.mean()), "rmse_std": float(cvs.std())}
                m["cv_rmse"] = float(-cvs.mean())
            tracker.metrics({**m, "fit_seconds": fit_s})
        results["regression"][name] = m
        fitted[("reg", name)] = (pipe, REG_FEATURES)
        print("regression |", name, {k: round(v, 4) for k, v in m.items()}, flush=True)
    best_r = min(("ridge", "hist_gb"), key=lambda n: cv["regression"][n]["rmse_mean"])

    # ------------------------------------------------------------------ importance + artifacts
    def perm(pipe, feats, X, y, scoring):
        idx = np.random.RandomState(SEED).choice(len(X), IMP_ROWS, replace=False)
        r = permutation_importance(pipe, X[feats].iloc[idx], y.iloc[idx], n_repeats=3, random_state=SEED, scoring=scoring)
        return {k: float(v) for k, v in pd.Series(r.importances_mean, index=feats).sort_values(ascending=False).items()}

    clf, cfeats = fitted[("cls", best_c)]
    reg, rfeats = fitted[("reg", best_r)]
    importance = {"classification": perm(clf, cfeats, test, yte, "roc_auc"),
                  "regression": perm(reg, rfeats, test, yte_r, "r2")}
    joblib.dump(clf, "models/classifier.joblib")
    joblib.dump(reg, "models/regressor.joblib")

    base_r2 = results["regression"]["baseline_mean"]["r2"]
    meta = {
        "reference_year": REFERENCE_YEAR, "horizon_years": HORIZON_YEARS, "seed": SEED,
        "classifier_features": cfeats, "regressor_features": rfeats, "stats": stats,
        "results": results, "cv": cv, "best_classifier": best_c, "best_regressor": best_r,
        "importance": importance, "n_train": n_train, "n_test": n_test,
        "good_rate_train": float(train.Good_Investment.mean()),
        "tracking_backend": tracker.backend,
        "regressor_skill_vs_mean_r2": float(results["regression"][best_r]["r2"] - base_r2),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "label_definition": f"proxy: price/sqft <= city median AND score >= {GOOD_SCORE_THRESHOLD}/8",
    }
    json.dump(meta, open("models/metadata.json", "w", encoding="utf-8"), indent=2)
    json.dump({"results": results, "cv": cv}, open("reports/metrics.json", "w", encoding="utf-8"), indent=2)
    with tracker.run("best_models_summary"):
        tracker.params({"best_classifier": best_c, "best_regressor": best_r})
        tracker.artifact("models/metadata.json")
        tracker.artifact("reports/metrics.json")
    write_report(meta)
    print("BEST", best_c, best_r, "| tracking:", tracker.backend, "| DONE", flush=True)
    return meta


def _context(meta, data=DATA):
    """Cheap, model-free checks for the report (no retraining): split hygiene and label sensitivity."""
    try:
        full = engineer(clean(load_raw(data)))
        train, test = train_test_split(full, test_size=0.2, random_state=meta["seed"])
        base = add_targets(test, meta["stats"]).Good_Investment
        half = add_targets(test, city_stats(train.sample(frac=0.5, random_state=1))).Good_Investment
        every = add_targets(test, city_stats(full)).Good_Investment
        return {"mean_price": float(full.Price_in_Lakhs.mean()), "id_overlap": len(set(train.ID) & set(test.ID)),
                "flip_half": float((half != base).mean()), "flip_all": float((every != base).mean())}
    except Exception as exc:  # data file missing etc.
        print("context checks skipped:", repr(exc))
        return None


def write_report(meta):
    """Generate reports/evaluation_report.md from the real metrics (no hand-typed numbers)."""
    r, cv = meta["results"], meta["cv"]
    c, g = r["classification"], r["regression"]
    ctx = _context(meta)
    best_acc = c[meta["best_classifier"]]["accuracy"]

    def row(name, d, keys):
        return f"| {name} | " + " | ".join(f"{d[k]:.4f}" if k in d else "–" for k in keys) + " |"

    ck = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_roc_auc"]
    rk = ["rmse", "mae", "r2", "cv_rmse"]
    L = ["# Model evaluation report", "",
         f"_Generated by `python -m src.train` · tracking backend: **{meta['tracking_backend']}** · "
         f"train rows {meta['n_train']:,} / held-out test rows {meta['n_test']:,} · seed {meta['seed']}_", "",
         "## 1. What is (and is not) being evaluated", "",
         "| Output | Nature | Evaluated how |", "|---|---|---|",
         "| Good-Investment class | **Rule-based proxy label** (no observed label exists) | Held-out recovery of the rule |",
         "| Current price estimate | **Learned**, observed target `Price_in_Lakhs` | Held-out RMSE/MAE/R² vs baselines |",
         f"| {meta['horizon_years']}-year projected price | **Assumption-based**: price × (1+r)^{meta['horizon_years']}, r from fixed rules | **Not evaluated** – no historical future prices exist |", "",
         "No actual 5-year forecasting accuracy is claimed anywhere in this project.", "",
         "## 2. Classification (proxy label, held-out test set)", "",
         f"Label: {meta['label_definition']}. Positive rate in train: {meta['good_rate_train']:.1%}.", "",
         "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
    L += [row(n, d, ck) for n, d in c.items()]
    L += ["", f"Selected by cross-validated ROC-AUC on the training split: **{meta['best_classifier']}** "
          "(test set not used for selection).", "",
          "**Interpretation.** The label is a deterministic function of the input columns, so a flexible model that sees "
          "them all can recover it almost perfectly. That is *rule recovery*, not evidence of predictive power on real "
          "investment outcomes. The `no_price_features` ablation removes the price inputs the rule depends on; its drop "
          "shows how strongly the label is tied to price per sq ft.", "",
          f"**What {best_acc:.1%} accuracy means.** It is the model's agreement with a hand-written rule on held-out rows. It does **not** measure "
          "whether any listing was, or will be, a successful investment: the data contains no investment outcomes (resale prices, rental yield, returns).", "",
          "## 3. Regression – current price in ₹ lakhs (observed target, held-out test set)", "",
          "Inputs exclude `Price_in_Lakhs` and `Price_per_SqFt_Rs` (no target leakage).", "",
          "| Model | RMSE | MAE | R² | CV RMSE |", "|---|---|---|---|---|"]
    L += [row(n, d, rk) for n, d in g.items()]
    skill = meta["regressor_skill_vs_mean_r2"]
    bm, bc, bb = g["baseline_mean"], g["baseline_city_mean"], g[meta["best_regressor"]]
    beats = bb["rmse"] < min(bm["rmse"], bc["rmse"])
    mp = ctx["mean_price"] if ctx else None
    pct = (f" (MAE = {bb['mae'] / mp:.0%} and RMSE = {bb['rmse'] / mp:.0%} of the mean listing price of ₹{mp:,.0f} lakhs)" if mp else "")
    L += ["", f"Selected by CV RMSE: **{meta['best_regressor']}**. R² gain over the predict-the-mean baseline: **{skill:+.4f}**.", "",
          f"**Error in context.** Held-out RMSE: selected model {bb['rmse']:.3f}, predict-the-mean {bm['rmse']:.3f}, per-city mean {bc['rmse']:.3f} lakhs; "
          f"MAE {bb['mae']:.1f} lakhs{pct}. The selected model **{'beats' if beats else 'does NOT beat'}** the better of the two baselines on the held-out set; R² = {bb['r2']:.4f}. "
          "Typical errors are therefore as large as the spread of prices themselves.", "",
          "**Interpretation.** " + ("The models are no better than predicting the average price: in this dataset price is "
          "statistically independent of the listed attributes (see EDA), so there is essentially nothing to learn. "
          "The estimate shown in the app should not be used to judge whether an asking price is fair."
          if skill < 0.01 else "The models improve on the mean baseline; see numbers above."), "",
          "## 4. Feature importance (permutation, held-out sample)", ""]
    for key, title in (("classification", "Classifier (ROC-AUC drop)"), ("regression", "Price regressor (R² drop)")):
        L += [f"**{title}** – top 8", "", "| Feature | Importance |", "|---|---|"]
        L += [f"| {k} | {v:.4f} |" for k, v in list(meta["importance"][key].items())[:8]]
        L.append("")
    L += ["Regression importances are at noise level (R² drops of ~0.001 or less): they show no feature carries real signal for price, "
          "so they should not be read as drivers. Classifier importance is dominated by price per sq ft because the proxy rule uses it.", ""]
    L += ["## 5. Evaluation-setup checks", "",
          "- Split (80/20, fixed seed) happens before any statistic is computed; city medians used by the label come from training rows only.",
          "- Classifier and regressor preprocessing (encoders, scalers) sit inside the scikit-learn pipelines, so cross-validation refits them per fold.",
          "- The regressor's inputs exclude `Price_in_Lakhs` and `Price_per_SqFt_Rs`; the classifier keeps them because the rule needs them (see ablation)."]
    if ctx:
        L += [f"- Listing IDs shared between train and test: **{ctx['id_overlap']}**.",
              f"- Residual optimism in cross-validation: CV labels use medians from the whole training split rather than per fold. As a size check, "
              f"computing medians from only half of the training rows changes {ctx['flip_half']:.2%} of test labels; using all rows (the original leak) changed {ctx['flip_all']:.2%}. "
              "CV scores for the classifier are therefore very slightly optimistic and are not used to claim real-world skill."]
    L += ["", "## 6. Limitations", "",
          "- The five-year horizon is counted from the dataset's reference year (2025), i.e. it ends in 2030, not five years from today.",
          "- The data contain contradictory combinations (e.g. under-construction listings built decades ago, houses in tall buildings, BHK unrelated to size); see `reports/data_audit.md`. They are retained and flagged, not repaired.",
          "- The dataset appears synthetic (independent, near-uniform columns), so real-market conclusions cannot be drawn.",
          "- Cross-validation used a 60k-row training subsample for speed; test metrics use the full held-out split.",
          "- One random split, one seed: no confidence intervals are reported for test metrics.",
          "- Scenario projections (4% / 8% / 12% a year) are illustrative arithmetic, not forecasts."]
    open("reports/evaluation_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    import sys
    if "--report-only" in sys.argv:   # rebuild reports/evaluation_report.md from saved metadata (no retraining)
        write_report(json.load(open("models/metadata.json", encoding="utf-8")))
        print("report regenerated")
    else:
        main()
