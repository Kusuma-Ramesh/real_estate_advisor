"""Experiment tracking: MLflow when installed, otherwise a local JSON-lines log.

The fallback exists so training never fails on a machine without MLflow; the report records
which backend was used (``Tracker.backend``).
"""
import contextlib
import json
import os
import time

try:  # pragma: no cover - depends on environment
    import mlflow
except Exception:  # ImportError or broken install
    mlflow = None


class Tracker:
    def __init__(self, experiment, uri="sqlite:///mlflow.db", fallback="reports/run_log.jsonl"):
        self.experiment, self.fallback = experiment, fallback
        self.backend = "mlflow" if mlflow is not None else "jsonl-fallback"
        self._cur = None
        if mlflow is not None:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
        else:
            os.makedirs(os.path.dirname(fallback), exist_ok=True)
            open(fallback, "w", encoding="utf-8").close()

    @contextlib.contextmanager
    def run(self, name):
        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                yield self
        else:
            self._cur = {"experiment": self.experiment, "run": name, "params": {}, "metrics": {}, "artifacts": []}
            try:
                yield self
            finally:
                with open(self.fallback, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._cur) + "\n")
                self._cur = None

    def params(self, d):
        d = {k: str(v) for k, v in d.items()}
        mlflow.log_params(d) if mlflow else self._cur["params"].update(d)

    def metrics(self, d):
        d = {k: float(v) for k, v in d.items()}
        mlflow.log_metrics(d) if mlflow else self._cur["metrics"].update(d)

    def artifact(self, path):
        mlflow.log_artifact(path) if mlflow else self._cur["artifacts"].append(path)
