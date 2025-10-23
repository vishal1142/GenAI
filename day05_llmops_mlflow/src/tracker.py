import mlflow
def init_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["uri"])
    mlflow.set_experiment(cfg.get("name","default"))
    if cfg.get("autolog",True):
        import transformers; mlflow.transformers.autolog()
