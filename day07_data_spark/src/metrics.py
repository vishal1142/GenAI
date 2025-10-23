# metrics.py — profile columns, produce tiny histogram-based drift signals
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from .utils import log, timed

def _histogram(df: DataFrame, colname: str, bins: int = 10) -> Dict[str, float]:
    # Approx histogram returns bin edges and counts
    edges, counts = df.selectExpr(f"approx_percentile({colname}, sequence(0, 1, 1.0/{bins})) as q").first()[0], []
    # Use edges to bucketize; here we just return quantiles as a compact profile
    return {"quantiles": [float(x) for x in edges]}

@timed
def profile_and_compare(df: DataFrame, ref_path: str, out_path: str) -> Dict[str, float]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    profile = {}
    for c in numeric_cols:
        try:
            profile[c] = _histogram(df, c, bins=10)
        except Exception:
            pass

    # Save current profile
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    # If no reference, write this as baseline and return no-drift
    ref_file = Path(ref_path)
    if not ref_file.exists():
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        log("Wrote initial reference profile (no drift by definition).")
        return {"drift": 0.0}

    # Compare very simply: average L1 distance of quantiles
    ref = json.loads(open(ref_file, encoding="utf-8").read())
    keys = set(profile.keys()).intersection(ref.keys())
    if not keys:
        return {"drift": 0.0}
    total, n = 0.0, 0
    for k in keys:
        p = profile[k]["quantiles"]; r = ref[k]["quantiles"]
        m = min(len(p), len(r))
        total += sum(abs(p[i] - r[i]) for i in range(m)) / m
        n += 1
    drift = (total / n) if n else 0.0
    log(f"Drift score (quantile L1 avg): {drift:.4f}")
    return {"drift": float(drift)}
