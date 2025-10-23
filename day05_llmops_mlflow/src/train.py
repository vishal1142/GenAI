from tqdm import tqdm
import torch, mlflow
from .model import train_step, eval_step
from .utils import timed

@timed
def run_training(model, tokenizer, loader, cfg):
    optim=torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with mlflow.start_run():
        for epoch in range(cfg["epochs"]):
            losses=[]
            for batch in tqdm(loader,desc=f"epoch {epoch+1}"):
                loss=train_step(model,batch,optim,device)
                losses.append(loss)
            mlflow.log_metric("train_loss",sum(losses)/len(losses))
        # quick eval on a few batches
        accs=[eval_step(model,b,device) for _,b in zip(range(5),loader)]
        mlflow.log_metric("eval_acc",sum(accs)/len(accs))
        mlflow.pytorch.log_model(model,"model")
