import torch
import torch.nn as nn
from . import config

class FakeNewsClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fc_layer = nn.Linear(config.EMBED_DIM, config.NUM_CLASSES)

    def forward(self, final_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            final_embedding (torch.Tensor): SentimentGuidedAttention을 통과한
                                            최종 임베딩 (Batch, Seq_Len, Dim)
        Returns:
            torch.Tensor: 각 클래스에 대한 로짓(logits) (Batch, Num_Classes)
        """
        pooled_output = torch.mean(final_embedding, dim=1)
        logits = self.fc_layer(pooled_output)
        
        return logits


