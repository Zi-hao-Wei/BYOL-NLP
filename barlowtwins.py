from multiprocessing import pool
from unittest import load_tests
import torch
import torch.nn as nn
import random
import math
import copy
import logging
from icecream import ic
from transformers import (
    BertPreTrainedModel,
    BertTokenizer,
)
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertEncoder
)
from mocose_tools import PATH_NOW
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from maskBatchNorm import MaskBatchNorm
logger = logging.getLogger(__name__)
from powerNorm import MaskPowerNorm
lambd=0.0051


class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay,total_step=15000):
        super().__init__()
        self.decay = decay
        self.total_step = total_step
        self.step = 0
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        self.step = self.step+1
        decay_new = 1-(1-self.decay)*(math.cos(math.pi*self.step/self.total_step)+1)/2
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(decay_new * e + (1. - decay_new) * m)

class PoolerWithoutActive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.hidden_size, config.out_size)
        self.layernorm1=nn.LayerNorm(config.out_size)
        self.layernorm2=nn.LayerNorm(config.out_size)

        self.l2 = nn.Linear(config.out_size, config.out_size)
        self.l3 = nn.Linear(config.out_size, config.out_size)

        self.activation=nn.ReLU()


        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, outputs, attention_mask, pooler_type="cls"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        attention_mask=attention_mask.squeeze()
        if pooler_type in ['cls_before_pooler', 'cls']:
            pooled = last_hidden[:, 0]
        elif pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif pooler_type == "avg_first_last":
            first_hidden = hidden_states[0].permute(0,2,1)
            last_hidden = hidden_states[-1].permute(0,2,1)
            pooled_result = (first_hidden.mean(dim=2) + last_hidden.mean(dim=2))/2
            print(pooled_result)
            return pooled_result
        elif pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
        x = self.l1(pooled)
        x = self.activation(x)
        x = self.layernorm1(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.layernorm2(x)
        x = self.l3(x)
        return x





def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def loss_fn(c): 
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss



class MoCoSEEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        for param in self.word_embeddings.parameters():
            param.requires_grad=False
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_drop_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0,
        sent_emb=False
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
    
        # drop out
        if not sent_emb:
            embeddings = self.dropout(embeddings)
        # embeddings = embeddings + torch.normal(0,0.1,size=embeddings.shape).cuda()
        return embeddings


class BarlowTwins(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.decay = config.ema_decay
        self.K = config.K
        self.K_start = config.K_start
        self.contextual_wordembs_aug = config.contextual_wordembs_aug
        self.online_embeddings = MoCoSEEmbeddings(config)
        self.online_encoder = BertEncoder(config)
        self.online_pooler = PoolerWithoutActive(config)
        self.bn = nn.BatchNorm1d(config.out_size, affine=False)
        self.loss_fct = loss_fn
        self.KLloss=nn.MSELoss()
        self.init_weights()
        self.prepare()

    def prepare(self):
        self.target_encoder = EMA(self.online_encoder,decay = self.decay)
        self.target_pooler = EMA(self.online_pooler,decay = self.decay)
        for params in self.target_encoder.parameters():
            params.requires_grad=False 
        for params in self.target_pooler.parameters():
            params.requires_grad=False 

    def save(self):
        torch.save(self.online_embeddings.state_dict(), "embeddings.pth")
        torch.save(self.online_encoder.state_dict(), "encoder.pth")
        torch.save(self.online_pooler.state_dict(), "pooler_dense.pth")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False# if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # print(input_ids.shape)

        if sent_emb:
            pass
        else:
            input_ids = input_ids[:,0]
            attention_mask = attention_mask[:,0]
            token_type_ids = token_type_ids[:,0]
                  
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)


        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        self.online_embeddings.eval()
        # Embedding
        view1 = self.online_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            #sent_emb=sent_emb
        )
        if sent_emb:
            attention_online = self.online_encoder(
                view1,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            cls_vec = self.online_pooler(attention_online,extended_attention_mask)
            # cls_vec = self.bn(cls_vec)
            attention_online.pooler_output = cls_vec
            return attention_online
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        self.target_encoder.update(self.online_encoder)
        self.target_pooler.update(self.online_pooler)
    
        self.online_embeddings.train()
        view2 = self.online_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )             
        
        # Encoder
       
        view_online_1 = self.online_encoder(
            view1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        view_online_2 = self.target_encoder(
            view2,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss2=F.cosine_similarity(view_online_1.last_hidden_state[:, 0],view_online_2.last_hidden_state[:, 0],dim=1)
        loss2=-loss2.mean()
        print(loss2)
        # 进行pooler
        p1 = self.online_pooler(view_online_1,extended_attention_mask)
        p2 = self.target_pooler(view_online_2,extended_attention_mask)

        p1 = self.bn(p1)
        p2 = self.bn(p2)
  
        c = p1.T @ p2
        c.div_(p1.shape[0])
        loss1 = self.loss_fct(c).mean()
        # set pooler output
        view_online_1.pooler_output = p1
        view_online_2.pooler_output = p2

        print(loss1)
        
        loss=loss1+loss2
        
   
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[p1,p2],
            attentions=None,
        )   
        

