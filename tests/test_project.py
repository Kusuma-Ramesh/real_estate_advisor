"""Run:  python -m unittest discover -s tests -v   (pytest also works)."""
import glob
import json
import os
import unittest

import pandas as pd

from src import features as F
from src.predict import assess, build_row, load_artifacts, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = "data/india_housing_prices.csv"
SAMPLE = dict(State="Maharashtra", City="Pune", Locality="Locality_490", Property_Type="Apartment", BHK=3,
              Size_in_SqFt=1500, Price_in_Lakhs=120.0, Year_Built=2012, Furnished_Status="Furnished", Floor_No=3,
              Total_Floors=10, Nearby_Schools=5, Nearby_Hospitals=5, Public_Transport_Accessibility="Medium",
              Parking_Space=True, Security=True, Amenities=["Gym", "Garden"], Facing="East", Owner_Type="Owner",
              Availability_Status="Ready_to_Move")


class Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA, nrows=20000)
        cls.df = F.engineer(F.clean(cls.raw))

    def test_no_target_leak_in_regression_features(self):
        self.assertFalse(set(F.PRICE_COLUMNS) & set(F.REG_FEATURES))
        self.assertTrue(set(F.PRICE_COLUMNS) <= set(F.CLS_FEATURES))
        self.assertNotIn("Locality", F.CLS_FEATURES + F.REG_FEATURES)

    def test_dates_consistent(self):
        self.assertTrue(((self.df.Age_of_Property + self.df.Year_Built) == F.REFERENCE_YEAR).all())

    def test_engineer_idempotent_and_clean_no_nans(self):
        again = F.engineer(self.df)
        pd.testing.assert_frame_equal(self.df, again)
        self.assertEqual(int(self.df[F.CLS_FEATURES].isna().sum().sum()), 0)

    def test_ppsf_in_rupees(self):
        r = self.df.iloc[0]
        self.assertAlmostEqual(r.Price_per_SqFt_Rs, r.Price_in_Lakhs * 1e5 / r.Size_in_SqFt)

    def test_floor_flag(self):
        self.assertTrue((self.df.Floor_Inconsistent == (self.df.Floor_No > self.df.Total_Floors)).all())
        self.assertLessEqual(self.df.Floor_Ratio.max(), 1.0)

    def test_labels_depend_only_on_supplied_stats(self):
        tr, te = self.df.iloc[:10000], self.df.iloc[10000:]
        stats = F.city_stats(tr)
        a = F.add_targets(te, stats)
        b = F.add_targets(te.sample(frac=1, random_state=0), stats).sort_index()
        pd.testing.assert_series_equal(a.Good_Investment, b.Good_Investment)
        # test-row prices must not change train-only stats
        self.assertEqual(stats, F.city_stats(tr))

    def test_growth_bounds_and_projection(self):
        s = F.city_stats(self.df)
        g = F.growth_rate(self.df, s)
        self.assertTrue(g.between(F.GROWTH_MIN, F.GROWTH_MAX).all())
        self.assertAlmostEqual(F.project_price(100, 0.08), 100 * 1.08 ** 5)


class DatasetValidation(unittest.TestCase):
    def test_missing_column_gives_clear_error(self):
        raw = pd.read_csv(DATA, nrows=50).drop(columns=["Size_in_SqFt"])
        with self.assertRaises(ValueError) as cm:
            F.clean(raw)
        self.assertIn("Size_in_SqFt", str(cm.exception))

    def test_nonpositive_rows_are_dropped_and_reported(self):
        import contextlib, io
        raw = pd.read_csv(DATA, nrows=50)
        raw.loc[0, "Size_in_SqFt"] = 0
        raw.loc[1, "Price_in_Lakhs"] = -5
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = F.clean(raw)
        self.assertEqual(len(out), 48)
        self.assertIn("2 non-positive", err.getvalue())
        self.assertTrue(F.engineer(out).Price_per_SqFt_Rs.notna().all())

    def test_real_dataset_needs_no_drops(self):
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            F.clean(pd.read_csv(DATA, nrows=20000))
        self.assertEqual(err.getvalue(), "")


class Predict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf, cls.reg, cls.meta = load_artifacts("models")

    def test_assess_ok(self):
        r = assess(SAMPLE, self.clf, self.reg, self.meta)
        self.assertEqual(r["errors"], [])
        self.assertTrue(0 <= r["good_probability"] <= 1)
        self.assertEqual(r["target_year"], F.REFERENCE_YEAR + F.HORIZON_YEARS)
        self.assertAlmostEqual(r["projected_price_lakhs"], F.project_price(120.0, r["assumed_growth_rate"]))
        self.assertEqual(set(r["scenarios"]), {"4%/yr", "8%/yr", "12%/yr"})

    def test_floor_validation(self):
        e, _ = validate(dict(SAMPLE, Floor_No=12))
        self.assertTrue(e)
        self.assertTrue(assess(dict(SAMPLE, Floor_No=12), self.clf, self.reg, self.meta)["errors"])

    def test_future_year_blocked_and_range_warning(self):
        self.assertTrue(validate(dict(SAMPLE, Year_Built=2030))[0])
        self.assertTrue(validate(dict(SAMPLE, Size_in_SqFt=100))[1])

    def test_unknown_city(self):
        self.assertTrue(assess(dict(SAMPLE, City="Atlantis"), self.clf, self.reg, self.meta)["errors"])

    def test_row_matches_training_columns(self):
        row = build_row(SAMPLE)
        self.assertFalse(set(self.meta["classifier_features"]) - set(row.columns))
        self.assertFalse(set(self.meta["regressor_features"]) - set(row.columns))

    def test_metadata_honesty_fields(self):
        m = self.meta
        self.assertNotIn("Price_in_Lakhs", m["regressor_features"])
        self.assertIn("baseline_mean", m["results"]["regression"])
        self.assertIn("baseline_majority_class", m["results"]["classification"])


class ProjectionAndEvaluation(unittest.TestCase):
    """Formula/unit checks for the assumption-based projection and reproduction of the saved metrics."""

    def test_projection_formula_and_units(self):
        raw = pd.read_csv(DATA, nrows=5000)
        df = F.engineer(F.clean(raw))
        s = F.city_stats(df)
        t = F.add_targets(df, s)
        expected = df.Price_in_Lakhs * (1 + F.growth_rate(df, s)) ** F.HORIZON_YEARS      # lakhs in, lakhs out
        pd.testing.assert_series_equal(t.Projected_Price_5Y_Lakhs, expected, check_names=False)
        self.assertTrue((t.Projected_Price_5Y_Lakhs > t.Price_in_Lakhs).all())          # growth rates are all positive
        self.assertTrue(t.Projected_Price_5Y_Lakhs.div(t.Price_in_Lakhs).between(
            (1 + F.GROWTH_MIN) ** 5, (1 + F.GROWTH_MAX) ** 5).all())
        self.assertAlmostEqual((df.Price_per_SqFt_Rs * df.Size_in_SqFt / 1e5 - df.Price_in_Lakhs).abs().max(), 0, places=6)

    def test_scenarios_ordered_and_labelled(self):
        clf, reg, meta = load_artifacts("models")
        r = assess(SAMPLE, clf, reg, meta)
        v = list(r["scenarios"].values())
        self.assertTrue(v[0] < v[1] < v[2])
        self.assertAlmostEqual(v[2] / v[0], (1 + F.GROWTH_MAX) ** 5 / (1 + F.GROWTH_MIN) ** 5)
        self.assertEqual(r["target_year"], F.REFERENCE_YEAR + F.HORIZON_YEARS)

    def test_plausibility_warnings(self):
        w = validate(dict(SAMPLE, Availability_Status="Under_Construction", Year_Built=1995))[1]
        self.assertTrue(any("Under-construction" in x for x in w))
        self.assertTrue(any("per BHK" in x for x in validate(dict(SAMPLE, Size_in_SqFt=500, BHK=5))[1]))
        self.assertEqual(validate(SAMPLE), ([], []))

    def test_saved_metrics_reproduce(self):
        """Recompute held-out metrics from the saved models and the same split; must match the saved metadata."""
        from sklearn.model_selection import train_test_split
        from src.train import cls_metrics, reg_metrics
        clf, reg, meta = load_artifacts("models")
        full = F.engineer(F.clean(pd.read_csv(DATA)))
        tr, te = train_test_split(full, test_size=0.2, random_state=meta["seed"])
        self.assertFalse(set(tr.ID) & set(te.ID))
        stats = F.city_stats(tr)
        self.assertEqual(stats, meta["stats"])
        te = F.add_targets(te, stats)
        c = cls_metrics(te.Good_Investment, clf.predict_proba(te[meta["classifier_features"]])[:, 1])
        r = reg_metrics(te.Price_in_Lakhs, reg.predict(te[meta["regressor_features"]]))
        sc = meta["results"]["classification"][meta["best_classifier"]]
        sr = meta["results"]["regression"][meta["best_regressor"]]
        self.assertAlmostEqual(c["accuracy"], sc["accuracy"], delta=1e-3)
        self.assertAlmostEqual(c["roc_auc"], sc["roc_auc"], delta=1e-3)
        self.assertAlmostEqual(r["rmse"], sr["rmse"], delta=0.05)
        self.assertAlmostEqual(r["r2"], sr["r2"], delta=1e-3)

    def test_regressor_never_claims_skill_it_lacks(self):
        _, _, meta = load_artifacts("models")
        rm = meta["results"]["regression"]
        floor = min(rm["baseline_mean"]["rmse"], rm["baseline_city_mean"]["rmse"])
        if rm[meta["best_regressor"]]["rmse"] >= floor:
            self.assertFalse(assess(SAMPLE, *load_artifacts("models"))["regressor_has_skill"])


class Outputs(unittest.TestCase):
    def test_20_eda_charts(self):
        self.assertEqual(len(glob.glob("eda_outputs/q[0-2][0-9]_*.png")), 20)
        self.assertTrue(os.path.exists("eda_outputs/findings.md"))

    def test_reports_exist(self):
        for p in ("reports/evaluation_report.md", "reports/metrics.json", "reports/data_audit.md"):
            self.assertTrue(os.path.exists(p), p)
        json.load(open("reports/metrics.json", encoding="utf-8"))

    def test_no_forecast_accuracy_claim(self):
        txt = open("reports/evaluation_report.md", encoding="utf-8").read()
        self.assertIn("Not evaluated", txt)
        self.assertIn("Error in context", txt)
        self.assertIn("successful investment", txt)


if __name__ == "__main__":
    unittest.main()
