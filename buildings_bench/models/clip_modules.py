import torch
from torch import nn
from transformers import DistilBertModel,  DistilBertTokenizer, BertTokenizer, BertModel
from buildings_bench.models.transformers import TimeSeriesSinusoidalPeriodicEmbedding

class PhysicsEncoder(nn.Module):
    """
    Encode physics input to a fixed size vector
    """
    def __init__(
        self,
        hidden_size = 128, 
        num_layers = 1, 
        pred_len = 168,
        trainable = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.pred_len    = pred_len
        self.hidden_size = hidden_size

        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.power_embedding = nn.Linear(1, 64)

        input_size = 103

        # bidirectional LSTM to encode exogenous input X
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # uni-directional LSTM to decode simulated measures Z
        self.decoder = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True)

        for p in self.parameters():
            p.requires_grad = trainable

    # get physics embedding
    def forward(self, batch):
        # exogenous input X
        x = torch.cat([
            self.day_of_year_encoding(batch['day_of_year']),
            self.day_of_week_encoding(batch['day_of_week']),
            self.hour_of_day_encoding(batch['hour_of_day']),
            batch["temperature"],
            batch["humidity"],
            batch["wind_speed"],
            batch["wind_direction"],
            batch["global_horizontal_radiation"],
            batch["direct_normal_radiation"],
            batch["diffuse_horizontal_radiation"]
        ], dim=2)

        z = batch["load"]
        z = z.roll(1, 1) # shift load by 1
        z[:, 0, :] = 0   # set initial load to 0
        z_hidden = self.power_embedding(z)

        _, (h_n, c_n) = self.encoder(x)

        # take the average of hidden states for both directions (?)
        h_n = torch.cat([
            h_n[:self.num_layers, ...].unsqueeze(0),
            h_n[self.num_layers:, ...].unsqueeze(0)
        ], dim=0).mean(dim=0)
        c_n = torch.cat([
            c_n[:self.num_layers, ...].unsqueeze(0),
            c_n[self.num_layers:, ...].unsqueeze(0)
        ], dim=0).mean(dim=0)

        outs, _ = self.decoder(z_hidden, (h_n, c_n))
        return outs # average on the time dimension,

class TextEncoder(nn.Module):
    """
    Encode test input to a fixed size vector
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased", 
        trainable=True,
        max_length=200
    ):
        super().__init__()
        self.max_length = max_length
        self.models = {
            "distilbert-base-uncased": [
                DistilBertModel.from_pretrained("distilbert-base-uncased"),
                DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            ],
            "bert-base-uncased": [
                BertModel.from_pretrained("bert-base-uncased"),
                BertTokenizer.from_pretrained('bert-base-uncased')
            ],
            # add more models here if needed
        }
        assert model_name in self.models
        self.model, self.tokenizer = self.models[model_name]

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        if model_name == "distilbert-base-uncased":
            for name, param in self.model.named_parameters():
                if "transformer.layer.5" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                
        elif model_name == "bert-base-uncased":
            for name, param in self.model.named_parameters():
                if "encoder.layer.11" not in name or "pooler" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, captions):
        encoded_captions = self.tokenizer(captions, padding=True, truncation=True, max_length=self.max_length) 
        encoded_caption_items = {
            key: torch.tensor(values).to("cuda")
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