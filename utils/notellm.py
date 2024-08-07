from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch
from torch.amp import autocast

class NoteLLM(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        model_name = cfg.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token":"[PAD]"})
        self.tokenizer.add_tokens(["[EMB]"])
        self.model = AutoModel.from_pretrained(model_name,device_map=cfg.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.fc = nn.Linear(2048,128)
        self.cfg = cfg
    
    @autocast("cuda")
    def forward(self,**x):
        x["input_ids"] = x["input_ids"].reshape([-1,*x["input_ids"].shape[2:]])
        x['attention_mask'] = x['attention_mask'].reshape([-1,*x['attention_mask'].shape[2:]])
        batch_size = x['input_ids'].shape[0]
        out_index = x['attention_mask'].sum(dim=-1) - 1 
        out = self.model(**x)
        out =  out.last_hidden_state[torch.arange(batch_size).to(self.cfg.device),out_index]
        out = self.fc(out)
        return out
        
    def get_tokenizer(self):
        return self.tokenizer