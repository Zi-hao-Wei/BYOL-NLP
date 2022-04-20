import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer,BertLayer
from transformers import BertTokenizer
from lightly.loss import NTXentLoss
from lightly.models.modules import NNMemoryBankModule

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class SimSiamModel(nn.Module):
    """SimSiamModel"""

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(SimSiamModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        dim=768
        prev_dim=4096
        pred_dim=4096
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        # self.loss_fct = NTXentLoss()
        # self.memorybank=NNMemoryBankModule(size=8192)


        self.pooling = pooling
        self.projector= nn.Sequential(nn.Linear(dim, prev_dim, bias=False),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.BatchNorm1d(prev_dim),
                                        # nn.Dropout(0.1),
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.BatchNorm1d(prev_dim),
                                        # nn.Dropout(0.1),
                                        nn.Linear(prev_dim, prev_dim))
                                
        self.predictor = nn.Sequential(nn.Linear(prev_dim, pred_dim, bias=False),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.BatchNorm1d(pred_dim),
                                        # nn.Dropout(0.1),
                                        nn.Linear(pred_dim, pred_dim, bias=False),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.BatchNorm1d(pred_dim),
                                        # nn.Dropout(0.1),
                                        nn.Linear(pred_dim, pred_dim)) # output laye
        self.bn1=nn.BatchNorm1d(prev_dim)
        self.bn2=nn.BatchNorm1d(prev_dim)

    def forward(self, input_ids, attention_mask, token_type_ids,sent_emb=False):
        if sent_emb:
            _cls1,z1,p1=self.forward_single(input_ids, attention_mask, token_type_ids,eval=True)
            return z1.detach()

        _cls1,z1,p1=self.forward_single(input_ids, attention_mask, token_type_ids,eval=False)
        _cls2,z2,p2=self.forward_single(input_ids, attention_mask, token_type_ids,eval=False)


        bnz1=self.bn1(p1)
        bnz2=self.bn1(p2)
        # bnp1=self.bn2(p1)
        # bnp2=self.bn2(p2)


        c1 = bnz1.T @ bnz2
        c1.div_(z1.shape[0])

        # c2 = bnz2.T @ bnp1
        # c2.div_(z2.shape[0])
        # c2 = bnp1.T @ bnz2
        # c2.div_(p1.shape[0])
        return _cls1,_cls2,p1,p2,z1.detach(),z2.detach(),c1

    def forward_single(self, input_ids, attention_mask, token_type_ids,eval=True):
        if eval:
            self.bert=self.bert.eval()
            self.projector=self.projector.eval()
            self.predictor=self.predictor.eval()

        else:
            self.bert=self.bert.train()
            self.projector=self.projector.train()
            self.predictor=self.predictor.train()
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)

        if self.pooling == 'cls':
            _cls=out.last_hidden_state[:, 0]
            z1=self.projector(_cls)
            p1=self.predictor(z1)  # [batch, 768]
            return _cls,z1,p1
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def loss_fn_bar(c): 
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + 0.0051 * off_diag
    return loss

def loss_fn_kd(outputs, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 1
    T = 10
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) 

    return KD_loss



def simsiam_loss(p1,p2,z1,z2):
    criterion = nn.CosineSimilarity(dim=-1).cuda()
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    # loss = -(criterion(p2, z1).mean())

    return loss

def consloss(x1,x2):
    criterion = nn.CosineSimilarity(dim=-1).cuda()
    loss = -criterion(x1, x2).mean()
    # loss = -(criterion(p2, z1).mean())

    return loss
