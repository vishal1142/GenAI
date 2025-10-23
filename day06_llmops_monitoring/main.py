from __future__ import annotations
import argparse
from pathlib import Path
from src.utils import set_seed
from src.data_gen import generate_datasets
from src.monitor import monitor_drift
from src.alerts import alert_if_drift
import yaml, mlflow

def main():
    cfg=yaml.safe_load(open("src/config.yaml"))
    set_seed(cfg["monitor"]["window_size"])
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    data_dir=Path("data")
    generate_datasets(data_dir)

    drift=monitor_drift(cfg)
    alert_if_drift(drift, cfg["monitor"]["alert_threshold"])

if __name__=="__main__":
    main()
