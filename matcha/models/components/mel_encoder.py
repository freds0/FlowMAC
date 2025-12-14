import torch
import torch.nn as nn
from matcha.models.components.transformer import BasicTransformerBlock

class MelEncoder(nn.Module):
    def __init__(self, n_feats, hidden_dim, n_layers, n_heads, dropout=0.1):
        super().__init__()
        # Projeção inicial (1x1 Conv descrita no paper)
        self.pre_net = nn.Conv1d(n_feats, hidden_dim, kernel_size=1)
        
        # Sequência de blocos Transformer
        self.transformer = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim, 
                num_attention_heads=n_heads, 
                attention_head_dim=hidden_dim // n_heads,
                dropout=dropout,
                activation_fn="gelu" 
            ) for _ in range(n_layers)
        ])
        
        # O paper não menciona explicitamente uma projeção de saída no bloco, 
        # mas geralmente é necessária para estabilização antes da quantização.
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        """
        Args:
            x: [Batch, n_feats, Time]
            mask: [Batch, 1, Time]
        """
        # 1. Projeção Inicial
        x = self.pre_net(x)
        
        # 2. Transformer (espera [Batch, Time, Dim])
        x = x.transpose(1, 2)
        
        if mask is not None:
            mask = mask.squeeze(1).bool() # [Batch, Time]
            
        for layer in self.transformer:
            x = layer(x, attention_mask=mask)
            
        x = self.norm(x)
        
        # 3. Retorna ao formato [Batch, Dim, Time]
        x = x.transpose(1, 2)
        return x