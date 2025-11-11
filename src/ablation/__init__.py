"""
Ablation study models for different aggregation mechanisms.
"""

from .attention_mil_model import AttentionMILLightning
from .mean_pooling_model import MeanPoolingTransformerMILLightning
from .attention_pooling_model import AttentionPoolingTransformerMILLightning
from .cls_token_model import CLSTokenTransformerMILLightning
from .embedding_mean_model import EmbeddingMeanTransformerMILLightning

__all__ = ['AttentionMILLightning', 'MeanPoolingTransformerMILLightning', 'AttentionPoolingTransformerMILLightning', 'CLSTokenTransformerMILLightning', 'EmbeddingMeanTransformerMILLightning']
