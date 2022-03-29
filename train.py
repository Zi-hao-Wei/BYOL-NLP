from transformers import BertTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk #, load_dataset, Dataset
import pandas as pd
from torch.utils.data import DataLoader
import torch
# from simsiam import *
from mocose import *

from transformers import BertConfig
from mocose_tools import MoCoSETrainer, PATH_NOW
from transformers.trainer import TrainingArguments

import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 64
train_dataset = load_from_disk(PATH_NOW+"/wiki_for_sts_32")
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True)

config = BertConfig()
config.out_size=768
config.mlp_layers=3
config.proj_layers=2

config.fgsm = 5e-9
config.embedding_drop_prob = 0.1
config.hidden_dropout_prob=0.1
config.attention_probs_dropout_prob=0.1
config.token_drop_prob = 0
config.feature_drop_prob = 0
config.token_shuffle = False
config.contextual_wordembs_aug = False

config.age_test = False

config.K = 256
config.K_start = 128
config.ema_decay = 0.75


config.neg_queue_slice_span = 256 # euqal to batch size, won't work if age_test=False

with open(PATH_NOW+r'/bert-base-uncased-weights/vocab.txt','r',encoding='utf8') as f:
    test_untokenizer = f.readlines()
untokenizer = [i[:-1] for i in test_untokenizer]
config.untokenizer = untokenizer

model = MoCoSEModel(config)
model.online_embeddings.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/embeddings.pth'))
model.online_encoder.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/encoder.pth'))
model.online_pooler.dense.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/pooler_dense.pth'))

model.prepare()
model = model.cuda()

non_optimizer_list = [model.target_encoder,model.target_pooler]
for layer in non_optimizer_list:
    for para in layer.parameters():
        para.requires_grad = False

def align_loss(x, y, alpha=2):    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def get_pair_emb(model, input_ids, attention_mask,token_type_ids):    
    outputs = model(input_ids = input_ids.cuda(),attention_mask=attention_mask.cuda(),token_type_ids=token_type_ids.cuda(),sent_emb=True)
    pooler_output = outputs.last_hidden_state[:,0]     
    #pooler_output = outputs.pooler_output
    z1, z2 = pooler_output[:batch_size], pooler_output[batch_size:]
    return z1.cpu(),z2.cpu()

def get_align(model, dataloader):
    align_all = []
    with torch.no_grad():        
        for data in dataloader:
            input_ids = torch.cat((data['input_ids'][0],data['input_ids'][1]))
            attention_mask = torch.cat((data['attention_mask'][0],data['attention_mask'][1]))
            token_type_ids = torch.cat((data['token_type_ids'][0],data['token_type_ids'][1]))

            z1,z2 = get_pair_emb(model, input_ids, attention_mask, token_type_ids)  
            z1 = F.normalize(z1,dim=1)
            z2 = F.normalize(z2,dim=1)
            align_all.append(align_loss(z1, z2))
            
    return align_all
    
def get_unif(model, dataloader):
    unif_all = []
    with torch.no_grad():        
        for data in dataloader:
            input_ids = torch.cat((data['input_ids'][0],data['input_ids'][1]))
            attention_mask = torch.cat((data['attention_mask'][0],data['attention_mask'][1]))
            token_type_ids = torch.cat((data['token_type_ids'][0],data['token_type_ids'][1]))
            z1,z2 = get_pair_emb(model, input_ids, attention_mask, token_type_ids)   
            #z = torch.cat((z1,z2))
            z = z1
            z = F.normalize(z,p=2,dim=1)
            unif_all.append(uniform_loss(z, t=2))
            
    return unif_all

args = TrainingArguments(
    output_dir = PATH_NOW+'/trained_models/mocose_base_out/',
    evaluation_strategy   = "steps",
    eval_steps            = 100,
    learning_rate         = 3e-4,
    # warmup_ratio=0.3,
    num_train_epochs      = 1.0,
    weight_decay          = 1e-6,
    per_device_train_batch_size = 256,
    per_device_eval_batch_size  = 256,
    dataloader_drop_last = True,
)
trainer = MoCoSETrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
