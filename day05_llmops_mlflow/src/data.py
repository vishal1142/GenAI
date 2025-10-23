# src/data.py — loads dataset and returns PyTorch DataLoader
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch


class SimpleDataset(Dataset):
    """Wraps tokenized text and labels into a PyTorch Dataset."""
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def get_dataloader(data_cfg, tokenizer, model_cfg):
    """
    Loads dataset and prepares a PyTorch DataLoader.
    data_cfg: dictionary from YAML -> data section
    model_cfg: dictionary from YAML -> model section (for max_length, batch_size)
    """
    ds = load_dataset(data_cfg["dataset"], split=data_cfg["split"])

    # Tokenize text column with model-based config
    enc = tokenizer(
        list(ds[data_cfg["text_col"]]),
        truncation=True,
        padding="max_length",
        max_length=model_cfg.get("max_length", 128)
    )

    dataset = SimpleDataset(enc, list(ds[data_cfg["label_col"]]))
    return DataLoader(dataset, batch_size=model_cfg.get("batch_size", 8), shuffle=True)
