from utils.kuairec import Kuai_Dataset
from utils.config import args
from torch.utils.data import DataLoader
from utils.notellm import NoteLLM
from utils.utils import train_one_epoch,create_optimizer_and_scheduler,eval_items,save_checkpoint,load_checkpoint
from torch.amp import GradScaler
import os

cfg = args.parse_args()
model = NoteLLM(cfg)
tokenizer = model.get_tokenizer()
train_set = Kuai_Dataset(cfg,tokenizer,train=True)
test_set = Kuai_Dataset(cfg,tokenizer,train=False)
train_loader = DataLoader(train_set,batch_size=cfg.batch_size_train,shuffle=True)
test_loader = DataLoader(test_set,batch_size=cfg.batch_size_test)

num_train_optimization_steps = int(len(train_loader) / cfg.gradient_accumulation_steps) * cfg.epoch_num
optimizer, scheduler= create_optimizer_and_scheduler(model,num_train_optimization_steps,cfg)
scaler = GradScaler("cuda")
resume_epoch =0
if cfg.resume:
    print("Loading checkpoint......")
    resume_epoch = load_checkpoint(model,optimizer,scaler,cfg)+1
    print("Load checkpoint succeed!")

model.to(cfg.device)

if not os.path.exists(f'checkpoint/log.txt'):
    file = open(f'checkpoint/log.txt',"w+")
    file.close()
for epoch in range(resume_epoch,cfg.epoch_num):
    print(f"==========epoch:{epoch}==========")
    train_one_epoch(model,train_loader,optimizer,scheduler,scaler,cfg)
    if (epoch+1 ) % cfg.num_eval == 0:
        res = eval_items(model,test_loader,cfg)
        metric  = cfg.metric_list
        out_print = f"Epoch: {epoch}  "
        for i,m in enumerate(metric):
            tt = round(float(res[i]),4)
            out_print +=f"Recall@{m}: {tt}  "
        print(out_print)
        with open(f'checkpoint/log.txt',"a") as f:
            f.write(out_print+"\n")
        save_checkpoint(model,optimizer,scaler,epoch)
