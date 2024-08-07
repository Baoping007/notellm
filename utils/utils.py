import torch
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .loss import contrastive_loss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.amp import autocast

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, (1 - float(current_step) / float(max(1, num_training_steps)))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_one_epoch(model, dataloader, optimizer, scheduler,scaler, cfg):
    model.train()
    pbar = tqdm(total=len(dataloader))
    for step, batch in enumerate(dataloader):
        if batch["input_ids"].shape[0]==1:
            continue
        for k, v in batch.items():
            batch[k] = v.to(cfg.device)
            
        with autocast("cuda"):
            out = model(**batch)
            # print(out.shape)
        with torch.no_grad():
            positive_paris = torch.arange(out.shape[0]).reshape(-1,2)
            negative_paris = torch.arange(out.shape[0]).reshape(-1,2)
            for i in range(negative_paris.shape[0]):
                choice_list = [j for j in range(out.shape[0]) if j not in [ negative_paris[i][1], negative_paris[i][0]]]
                negative_paris[i][1] = np.random.choice(choice_list)
            positive_paris.to(cfg.device)
            negative_paris.to(cfg.device)
        with autocast("cuda"):
            loss = contrastive_loss(out,positive_paris,negative_paris,cfg.temperature)

        
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        # loss.backward()
        scaler.scale(loss).backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Update learning rate schedule
            scheduler.step()
        pbar.update(1)
        with torch.no_grad():
            l = round(loss.item(),5)
        pbar.set_postfix(loss=f'{l}')
            
def eval_items(model,dataloader,cfg):
    model.eval()
    embeddings = None
    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Eval')):
        for k, v in batch.items():
            batch[k] = v.to(cfg.device)
        with torch.no_grad():
            out = model(**batch)
            if embeddings == None:
                embeddings = out
            else:
                embeddings = torch.cat((embeddings,out),dim=0)
    # print(f"embedding shape: {embeddings.shape}")
    labels = np.load(cfg.test_data_path)[:,1:]
    # print(f"Items length: {labels.shape}")
    
    embeddings = embeddings.cpu().numpy()
    items_items_matrix = cosine_similarity(embeddings,embeddings)
    embeddings = items_items_matrix.argsort(axis=-1)[:,::-1][:,1:]
    # np.save("tmp.npy",embeddings)

    metric_list = cfg.metric_list
    results = np.array([0] *len(metric_list),dtype=np.float32)
    for i in tqdm(range(len(labels))):
        for cnt in range(len(metric_list)):
            a = labels[i,:metric_list[cnt]]
            b = embeddings[i,:metric_list[cnt]]
            results[cnt] +=np.intersect1d(a,b).shape[0]/metric_list[cnt]
    results = results/len(labels)
    return results
    

def create_optimizer_and_scheduler(model: nn.Module, num_train_optimization_steps, cfg):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler


def save_checkpoint(model,optimizer,scaler,epoch):
    checkpoint = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
    torch.save(checkpoint, f"checkpoint/{epoch}.pth")

def load_checkpoint(model,optimizer,scaler,cfg):
    checkpoint = torch.load(cfg.checkpoint_path,map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"]