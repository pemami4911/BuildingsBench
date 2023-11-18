import torch
from torch import nn
from transformers import DistilBertModel,  DistilBertTokenizer, BertTokenizer, BertModel
from buildings_bench.models.transformers import TimeSeriesSinusoidalPeriodicEmbedding

class encoder_x(nn.Module):
    """
    Encode exogenous simulation input X to a fixed size vector
    """
    def __init__(
        self,
        hidden_size = 128, 
        num_layers = 1, 
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)

        input_size = 103 

        # bidirectional LSTM 
        self.encoder = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)


    def forward(self, batch):
        # [batch_size, seq_len, 103]
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
        
        outs, (h_n, c_n) = self.encoder(x)

        # take the average of hidden states for both directions to get an embedding for x
        x_embedding = torch.cat([
            outs[:,:,:self.hidden_size].unsqueeze(3),
            outs[:,:,self.hidden_size:].unsqueeze(3)
        ], dim=3).mean(dim=3)
        
        return x_embedding 

class encoder_yhat(nn.Module):
    """
    Encode attribute caption Y_hat to a fixed size vector
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased", 
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


class encoder_z(nn.Module):
    """
    Encode simulation output Z to a fixed size vector
    """
    def __init__(
        self,
        hidden_size = 128, 
        num_layers = 1, 
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.power_embedding = nn.Linear(1, 64)

        # (unidirectional) LSTM 
        self.encoder = nn.LSTM(64, self.hidden_size, num_layers=self.num_layers, batch_first=True)
  
    def forward(self, z):
        z = z.roll(1, 1) # shift load by 1
        z[:, 0, :] = 0   # set initial load to 0
        z_hidden = self.power_embedding(z)

        outs, (hn, cn)  = self.encoder(z_hidden)
        
        return outs

