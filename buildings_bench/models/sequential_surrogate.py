import torch
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sequential_surrogate_modules import encoder_x, encoder_yhat, encoder_z
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

'''
Dynamics MLP 
'''

class sequentialsurrogateMLP(BaseModel):

    def __init__(self, 
                hidden_dim = 256,
                context_len=1,
                pred_len=1,
                lstm_hidden_size = 128,
                lstm_num_layers = 1,
                text_encoder_name = "distilbert-base-uncased",
                text_max_length = 200,
                continuous_head='mse',
                continuous_loads=True):
        super(sequentialsurrogateMLP,self).__init__(context_len, pred_len, continuous_loads)
        self.continuous_head = continuous_head
        out_dim = 1 if self.continuous_head == 'mse' else 2
        self.logits = nn.Linear(hidden_dim, out_dim)
        self.pred_len = pred_len
        
        self.encoder_x = encoder_x(
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers
        )
        self.encoder_yhat = encoder_yhat(
            model_name = text_encoder_name,
            max_length = text_max_length
        )
        self.encoder_z = encoder_z(
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size*2 + 768, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        self.train_mode = True

    def forward(self, x):
        ht = self.encoder_x(x)
        g = self.encoder_yhat(x["building_description"]) 
        g = g.unsqueeze(1).repeat(1, self.pred_len, 1) ## to get g in the shape [batch, pred_len, embedding_length]
        qt_prev = self.encoder_z(x["load"])
        
        input_embedding_to_mlp = torch.cat((ht, g, qt_prev), dim = 2)
        output_embedding = self.mlp(input_embedding_to_mlp)
        return self.logits(output_embedding)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x)
        return out, None
    
    def loss(self, x, y):
        if self.continuous_head == 'mse':
            return F.mse_loss(x, y)
        elif self.continuous_head == 'gaussian_nll':
            return F.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y,
                                       F.softplus(x[:, :, 1].unsqueeze(2)) **2)

    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True    
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        return None