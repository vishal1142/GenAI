# strategies.py — Strategy pattern for drift detection
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
import pandas as pd, numpy as np

class EvidentlyDrift:
    def __init__(self, report_dir):
        self.report_dir=report_dir
    def detect(self, ref:pd.DataFrame, cur:pd.DataFrame):
        rep=Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=ref, current_data=cur)
        rep.save_html(str(self.report_dir/"drift_report.html"))
        result=rep.as_dict()
        return result["metrics"][0]["result"]["dataset_drift"]

class CustomDrift:
    """Custom drift using mean/std difference heuristic."""
    def detect(self, ref:pd.DataFrame, cur:pd.DataFrame):
        diffs=[]
        for col in ref.columns:
            if col=="prediction": continue
            d=abs(ref[col].mean()-cur[col].mean())/max(ref[col].std(),1e-9)
            diffs.append(d)
        score=float(np.mean(diffs))
        return score>0.5

def make_drift_strategy(name:str,report_dir):
    name=(name or "evidently").lower()
    if name=="evidently": return EvidentlyDrift(report_dir)
    if name=="custom": return CustomDrift()
    raise ValueError("Unknown drift strategy")
