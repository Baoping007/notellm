from transformers import LlamaTokenizer
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class Kuai_Dataset(Dataset):
    def __init__(self,cfg,tokenizer,train=True):
        self.cfg = cfg
        self.train = train
        self.tokenizer = tokenizer
        if train:
            self.items_pairs = np.load(cfg.train_data_path)
        else:
            self.items_pairs = np.load(cfg.test_data_path)
        
        self.items_index2id = self.items_pairs[:,0]

        self.captions = pd.read_csv(cfg.captions_path)
        # self.captions = pd.read_csv(cfg.captions_path,lineterminator='\n')
    
    def __getitem__(self,index):
        items = self.items_pairs[index]
        # 被匹配正样本ID
        item_id = items[0]
        item_positive =self.items_index2id[np.random.choice(items[1:21])]
        
        #构建token
        input_ids,attention_mask = self.build_token(item_id)
        if not self.train:
            return {"input_ids":torch.tensor([input_ids]),"attention_mask":torch.tensor([attention_mask])}

        input_ids_positive,attention_mask_positive = self.build_token(item_positive)
        return {"input_ids":torch.tensor([input_ids,input_ids_positive]),"attention_mask":torch.tensor([attention_mask,attention_mask_positive])}
        
    def __len__(self):
        return self.items_pairs.shape[0]
        # return self.items_pairs.shape[0]//100
    
    def build_prompt(self,item_id):
        item = self.captions[self.captions["video_id"]==item_id]
        caption = str(item['caption'].values[0])
        topic_tag = str(item['topic_tag'].values[0])
        first_level_category_name = str(item['first_level_category_name'].values[0])
        second_level_category_name = str(item['second_level_category_name'].values[0])
        third_level_category_name = str(item['third_level_category_name'].values[0])
        caption = " " if caption=="nan" else caption
        topic_tag = " " if topic_tag=="nan" else topic_tag
        first_level_category_name = " " if first_level_category_name=="nan" else first_level_category_name
        second_level_category_name = " " if second_level_category_name=="nan" else second_level_category_name
        third_level_category_name = " " if third_level_category_name=="nan" else third_level_category_name
        
        input_prompt = ["Extract information in json format and compress it into a word suitable for recommendation system.{'first level category name': '",first_level_category_name,"', 'second level category name': '",second_level_category_name,"', 'third level category name': '",third_level_category_name,"','topic tag': '",topic_tag,"','caption': '",caption,"'}，The compressed word is: [EMB]"]
        
        mask_prompt = [0,self.cfg.max_category_len,0,self.cfg.max_category_len,0,self.cfg.max_category_len,0,self.cfg.max_topic_len,0,self.cfg.max_caption_len,0]
    
        return input_prompt,mask_prompt
    
    def build_token(self,item_id):
        input_prompt,mask_prompt = self.build_prompt(item_id)
        input_ids = [self.tokenizer.bos_token_id]
        # print("============")
        # print(len(input_prompt))
        for i,mask in zip(input_prompt,mask_prompt):
            tmp = self.tokenizer(i)["input_ids"][1:]
            # print(len(tmp))
            if mask>0 and len(tmp)>mask:
                tmp = tmp[:mask]
            input_ids+= tmp
        attention_mask = [1]*len(input_ids)
        PAD_LEN = self.cfg.max_token_len-len(input_ids)
        if PAD_LEN<=0:
            input_ids = input_ids[:self.cfg.max_token_len]
            attention_mask = attention_mask[:self.cfg.max_token_len]
        else:
            tmp = [self.tokenizer.pad_token_id]*PAD_LEN
            input_ids = input_ids+tmp
            tmp = [0]*PAD_LEN
            attention_mask = attention_mask+tmp
        return input_ids,attention_mask
        
        
