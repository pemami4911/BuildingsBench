import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig,  DistilBertTokenizer


class PhysicsEncoder(nn.Module):
    """
    Encode physics input to a fixed size vector
    """
    pass

class TextEncoder(nn.Module):
    """
    Encode test input to a fixed size vector
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased", 
        pretrained=True, 
        trainable=True,
        max_length = 200
    ):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, captions):
        encoded_captions = self.tokenizer(captions, padding=True, truncation=True, max_length=max_length) 
        encoded_caption_items = {
            key: torch.tensor(values[idx])
            for key, values in encoded_captions.items()
        }
        output = self.model(input_ids=encoded_caption_items["input_ids"], attention_mask=encoded_caption_items["attention_mask"])
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x