from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk #, load_dataset, Dataset
import pandas as pd
from torch.utils.data import DataLoader
import torch
from simsiam import *
from transformers import BertConfig
from mocose_tools import MoCoSETrainer, PATH_NOW, evalModel
from transformers.trainer import TrainingArguments

import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 256
train_dataset = load_from_disk(PATH_NOW+"/wiki_for_sts_32")
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True)

config = BertConfig()
config.out_size=768
config.hidden_size=768
config.mlp_layers=2
config.proj_layers=2
config.fgsm = 5e-9
config.embedding_drop_prob = 0.2
config.hidden_dropout_prob=0.2
config.attention_probs_dropout_prob=0.2
config.token_drop_prob = 0
config.feature_drop_prob = 0
config.token_shuffle = False
config.contextual_wordembs_aug = False
config.age_test = False
config.K = 256
config.K_start = 128
config.ema_decay = 0.95
config.age_test=False

config.neg_queue_slice_span = 256 # euqal to batch size, won't work if age_test=False
path1="/trained_models/mocose_base_out/best-model"
with open(PATH_NOW+r'/bert-base-uncased-weights/vocab.txt','r',encoding='utf8') as f:
    test_untokenizer = f.readlines()
untokenizer = [i[:-1] for i in test_untokenizer]
config.untokenizer = untokenizer

# model=BertModel.from_pretrained("bert-base-uncased")
# model.save()

# model = MoCoSEModel(config)
# model.online_embeddings.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/embeddings.pth'))
# model.online_encoder.load_state_dict(torch.load(PATH_NOW+'/bert-base-uncased-weights/encoder.pth'))

model = MoCoSEModel.from_pretrained(PATH_NOW+path1)

# model2 = BarlowTwins.from_pretrained(PATH_NOW+path2)
# model1.save()
# model = BarlowTwins(config=config)
# for parm in model1.

model = model.cuda()
# # model2 = model2.cuda()

tokenizer = BertTokenizer.from_pretrained(PATH_NOW+path1)

sum_acc = evalModel(model,tokenizer, pooler = 'cls_after_pooler')
