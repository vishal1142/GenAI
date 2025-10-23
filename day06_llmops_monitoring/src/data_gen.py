# data_gen.py — simulate reference & current datasets (e.g., embeddings or features)
import numpy as np, pandas as pd
from pathlib import Path
from .utils import ensure_dir

def generate_datasets(out_dir:Path):
    ensure_dir(out_dir)
    n=1000
    ref=pd.DataFrame({
        "feature1": np.random.normal(0,1,n),
        "feature2": np.random.normal(5,2,n),
        "prediction": np.random.binomial(1,0.5,n)
    })
    # current with slight drift
    cur=pd.DataFrame({
        "feature1": np.random.normal(0.2,1.2,n),
        "feature2": np.random.normal(6,2.5,n),
        "prediction": np.random.binomial(1,0.55,n)
    })
    ref.to_csv(out_dir/"reference.csv",index=False)
    cur.to_csv(out_dir/"current.csv",index=False)
    print("Generated synthetic reference & current data.")
