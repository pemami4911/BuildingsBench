import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import os
from buildings_bench.models.clip_modules import PhysicsEncoder, TextEncoder, ProjectionHead

'''
CLIP 
'''
class CLIP(nn.Module):
    def __init__(
        self,
        physics_embedding = 512,
        num_layers = 2,
        pred_len = 168,
        text_embedding = 768,
        text_encoder_name = "distilbert-base-uncased",
        trainable = True, 
        max_length = 200,
        projection_dim = 128,
        continuous_loads = True,
        temperature = 1.0
    ):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(
            pred_len = pred_len,
            hidden_size = physics_embedding, 
            num_layers = num_layers,
            trainable = True)
        self.text_encoder = TextEncoder(
            model_name = text_encoder_name, 
            trainable = trainable, 
            max_length = max_length)
        self.physics_projection = ProjectionHead(embedding_dim=physics_embedding, projection_dim = projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding, projection_dim = projection_dim)
        self.temperature = temperature
        self.continuous_loads = continuous_loads
        self.pred_len = pred_len

    def forward(self, x):
        captions = x["building_description"]

        # Getting Physics-based and Text Features
        physics_features = self.physics_encoder(x)
        physics_features = physics_features.mean(dim=1) # average on time dimension
        text_features = self.text_encoder(captions)

        # Getting Physics-based and Text Embeddings (with same dimension)
        physics_embeddings = self.physics_projection(physics_features)
        text_embeddings = self.text_projection(text_features)

        return text_embeddings, physics_embeddings
    
    def loss(self, text_embeddings, physics_embeddings):
        # Calculating the Loss
        logits = (text_embeddings @ physics_embeddings.T) / self.temperature
        physicss_similarity = physics_embeddings @ physics_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (physicss_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        physicss_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (physicss_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def load_from_checkpoint(self, checkpoint_path):
        return None
        
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()