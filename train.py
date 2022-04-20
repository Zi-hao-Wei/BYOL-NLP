import os

from Simbarlow import SimBarlow
os.environ["WANDB_DISABLED"] = "True"

from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk #, load_dataset, Dataset
import pandas as pd
from torch.utils.data import DataLoader
import torch
# from simsiam import *
import random

from transformers import BertConfig
from mocose_tools import MoCoSETrainer, PATH_NOW
from transformers.trainer import TrainingArguments

import torch.nn.functional as F
wandb = None

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 256
train_dataset = load_from_disk(PATH_NOW+"/wiki_for_sts_32_2")
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True, shuffle=True)

config = BertConfig()
config.out_size=4096
config.hidden_size=768
config.mlp_layers=2
config.proj_layers=2
config.fgsm = 5e-9
config.embedding_drop_prob = 0.05
config.hidden_dropout_prob=0.15
config.attention_probs_dropout_prob=0.15
config.token_drop_prob = 0
config.feature_drop_prob = 0
config.token_shuffle = False
config.contextual_wordembs_aug = False
config.output_hidden_states = False
config.K = 256
config.K_start = 128
config.ema_decay = 0.995
config.age_test=False

config.neg_queue_slice_span = 256 # euqal to batch size, won't work if age_test=False

with open(PATH_NOW+r'/bert-base-uncased-weights/vocab.txt','r',encoding='utf8') as f:
    test_untokenizer = f.readlines()
untokenizer = [i[:-1] for i in test_untokenizer]
config.untokenizer = untokenizer
# random.seed(0)
model = SimBarlow(config)
model.online_embeddings.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/embeddings.pth'))
model.online_encoder.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/encoder.pth'))
# model.online_pooler.dense.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/pooler_dense.pth'))

model = model.cuda()

args = TrainingArguments(
    output_dir = PATH_NOW+'/trained_models/mocose_base_out/',
    evaluation_strategy   = "steps",
    eval_steps            = 50,
    learning_rate         = 1e-5,
    num_train_epochs      = 1.0,
    weight_decay          = 1e-7,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    dataloader_drop_last = True,
)


trainer = MoCoSETrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
