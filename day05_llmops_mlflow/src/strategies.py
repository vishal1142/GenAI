# strategies.py – Strategy pattern for model backend selection
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, mlflow

def make_model(name:str, strategy:str="local"):
    """
    Choose model-loading strategy.
    local: standard AutoModel
    hf_hub: uses from_pretrained with trust_remote_code
    """
    if strategy=="local":
        tok=AutoTokenizer.from_pretrained(name)
        model=AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        return model, tok
    elif strategy=="hf_hub":
        tok=AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model=AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True, num_labels=2)
        return model, tok
    else:
        raise ValueError("Unknown strategy")
