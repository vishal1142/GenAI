import pandas as pd, mlflow
from pathlib import Path
from .strategies import make_drift_strategy
from .utils import timed, ensure_dir

@timed
def monitor_drift(cfg:dict):
    ref=pd.read_csv(cfg["data"]["ref_file"])
    cur=pd.read_csv(cfg["data"]["new_file"])
    report_dir=Path(cfg["monitor"]["report_dir"])
    ensure_dir(report_dir)
    strat=make_drift_strategy(cfg["monitor"]["drift_strategy"],report_dir)
    with mlflow.start_run(run_name="drift_check"):
        result=strat.detect(ref,cur)
        mlflow.log_param("strategy",cfg["monitor"]["drift_strategy"])
        mlflow.log_artifact(str(report_dir/"drift_report.html"))
        if isinstance(result,bool): drift=result
        else: drift=bool(result)
        mlflow.log_metric("drift_detected",float(drift))
        print(f"Drift detected? {drift}")
    return drift
