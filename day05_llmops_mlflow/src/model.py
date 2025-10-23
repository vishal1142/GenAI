import torch, torch.nn.functional as F
def train_step(model,batch,optim,device):
    model.train()
    batch={k:v.to(device) for k,v in batch.items()}
    out=model(**batch)
    loss=out.loss
    loss.backward()
    optim.step(); optim.zero_grad()
    return float(loss.detach().cpu())

@torch.no_grad()
def eval_step(model,batch,device):
    model.eval()
    batch={k:v.to(device) for k,v in batch.items()}
    out=model(**batch)
    preds=out.logits.argmax(dim=-1).cpu()
    labels=batch["labels"].cpu()
    acc=(preds==labels).float().mean().item()
    return acc
