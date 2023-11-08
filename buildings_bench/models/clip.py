import torch
from torch import nn
import torch.nn.functional as F

from clip_modules import PhysicsEncoder, TextEncoder, ProjectionHead

'''
CLIP 
'''
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature = 1.0,
        physics_embedding = 768,
        text_embedding = 768,
        model_name = "distilbert-base-uncased",
        pretrained = True,
        trainable = True, 
        max_length = 200,
        projection_dim = 256
    ):
        super().__init__()
        self.physics_encoder = PhysicsEncoder()
        self.text_encoder = TextEncoder(
            model_name = model_name, 
            pretrained = pretrained, 
            trainable = trainable, 
            max_length = max_length,
            )
        self.physics_projection = ProjectionHead(embedding_dim=physics_embedding, projection_dim = projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding, projection_dim = projection_dim)
        self.temperature = temperature

    def forward(self, batch):

        caption = "" # Get text for the batch using building ids
        # Getting Physics-based and Text Features
        physics_features = self.physics_encoder(batch)
        text_features = self.text_encoder(captions)
        # Getting Physics-based and Text Embeddings (with same dimension)
        physics_embeddings = self.physics_projection(physics_features)
        text_embeddings = self.text_projection(text_features)

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


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()