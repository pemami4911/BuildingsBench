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
        temperature = 1.0,
        physics_embedding = 512,
        text_embedding = 768,
        text_encoder_name = "distilbert-base-uncased",
        pretrained = True,
        trainable = True, 
        max_length = 200,
        projection_dim = 128,
        continuous_loads = True,
        pred_len=1
    ):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(
            hidden_size = physics_embedding, 
            trainable = True)
        self.text_encoder = TextEncoder(
            model_name = text_encoder_name, 
            pretrained = pretrained, 
            trainable = trainable, 
            max_length = max_length,
            )
        self.physics_projection = ProjectionHead(embedding_dim=physics_embedding, projection_dim = projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding, projection_dim = projection_dim)
        self.temperature = temperature
        self.metadata = Path(os.environ.get('BUILDINGS_BENCH', '')) / "metadata_dev"
        self.datasets = ["comstock_tmy3", "resstock_tmy3", "comstock_amy2018", "resstock_amy2018"]
        self.continuous_loads = continuous_loads
        self.pred_len = pred_len

    def forward(self, x):
        captions = [] 
        for dataset_id, bldg_id in zip(x["dataset_id"], x["building_id"]):
            dataset_id = int(dataset_id.item())
            bldg_id = int(bldg_id.item())
            with open(self.metadata / "simcap" / self.datasets[dataset_id] / f"{bldg_id}_cap.txt", "r") as f:
                captions.append(f.read())

        # Getting Physics-based and Text Features
        physics_features = self.physics_encoder(x)
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