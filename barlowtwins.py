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

class PoolerWithoutActive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.l2 = nn.Linear(config.hidden_size, config.out_size)
        self.activation=nn.Tanh()


        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        x = self.l1(first_token_tensor)
        x=self.activation(x)
        x=self.l2(x)
        x=self.activation(x)

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
        self.init_weights()


    
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
            attention_online_last = attention_online[0]
            cls_vec = self.online_pooler(attention_online_last)
            attention_online.pooler_output = cls_vec
            return attention_online
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
       

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

        view_online_2 = self.online_encoder(
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


        # pooler
        attention_online_out_1 = view_online_1[0]
        attention_online_out_2 = view_online_2[0]
    
        # 进行pooler
        p1 = self.online_pooler(attention_online_out_1)
        p2 = self.online_pooler(attention_online_out_2)


        c = self.bn(p1).T @ self.bn(p2)
        c.div_(p1.shape[0])
       
        # set pooler output
        view_online_1.pooler_output = p1
        view_online_2.pooler_output = p2

        loss = self.loss_fct(c)
        loss=loss.mean()
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[p1,p2],
            attentions=None,
        )   
        

