"""Exercises src.tracking against a STUB mlflow (real MLflow is not installed in CI sandboxes without network).
This checks our call pattern only; it does not prove compatibility with a specific MLflow release."""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrackingTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("mlflow", None)
        import src.tracking
        importlib.reload(src.tracking)

    def test_fallback_jsonl(self):
        sys.modules["mlflow"] = None  # force ImportError path
        import src.tracking as T
        importlib.reload(T)
        with tempfile.TemporaryDirectory() as d:
            t = T.Tracker("exp", fallback=os.path.join(d, "log.jsonl"))
            self.assertEqual(t.backend, "jsonl-fallback")
            with t.run("r1"):
                t.params({"a": 1}); t.metrics({"m": 0.5}); t.artifact("x")
            rec = json.loads(open(os.path.join(d, "log.jsonl"), encoding="utf-8").read().splitlines()[0])
            self.assertEqual(rec["params"], {"a": "1"})
            self.assertEqual(rec["metrics"], {"m": 0.5})

    def test_mlflow_branch_calls(self):
        calls = []
        stub = types.ModuleType("mlflow")
        import contextlib
        stub.set_tracking_uri = lambda u: calls.append(("uri", u))
        stub.set_experiment = lambda e: calls.append(("exp", e))
        stub.log_params = lambda d: calls.append(("params", d))
        stub.log_metrics = lambda d: calls.append(("metrics", d))
        stub.log_artifact = lambda p: calls.append(("artifact", p))
        @contextlib.contextmanager
        def start_run(run_name=None):
            calls.append(("start", run_name)); yield
        stub.start_run = start_run
        sys.modules["mlflow"] = stub
        import src.tracking as T
        importlib.reload(T)
        t = T.Tracker("exp")
        self.assertEqual(t.backend, "mlflow")
        with t.run("r"):
            t.params({"a": 1}); t.metrics({"m": 1}); t.artifact("f")
        self.assertEqual([c[0] for c in calls], ["uri", "exp", "start", "params", "metrics", "artifact"])


if __name__ == "__main__":
    unittest.main()
