import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.utils.config import config

class BertEncoder(nn.Module):
    def __init__(self, model_name=None, freeze=True):
        super(BertEncoder, self).__init__()
        self.model_name = model_name or config.MODEL_NAME
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        if freeze:
            self.freeze_weights()
            
    def freeze_weights(self):
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def unfreeze_weights(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Returns mean-pooled embeddings.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # (B, SeqLen, Dim)
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
