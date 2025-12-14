import torch.nn as nn
# Certifique-se de ter instalado: pip install vector-quantize-pytorch
from vector_quantize_pytorch import ResidualVQ

class QuantizerWrapper(nn.Module):
    def __init__(self, dim, codebook_size=256, num_quantizers=8):
        super().__init__()
        # Configuração baseada no paper do FlowMAC:
        # "FlowMAC uses a codebook size of 256 and 8 quantizer stages"
        self.rvq = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            kmeans_init=True,   # Ajuda na estabilidade do treinamento
            kmeans_iters=10,
            threshold_ema_dead_code=2  # Ajuda a reviver códigos mortos
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor de entrada [Batch, Dim, Time] vindo do Encoder
        Returns:
            quantized: Tensor quantizado [Batch, Dim, Time]
            commit_loss: Loss de quantização
        """
        # A biblioteca vector-quantize-pytorch espera [Batch, Time, Dim]
        x = x.transpose(1, 2)
        
        # Realiza a quantização residual
        # indices são os códigos discretos (útil para compressão/inferência)
        quantized, indices, commit_loss = self.rvq(x)
        
        # Retorna ao formato original [Batch, Dim, Time] para o Decoder
        quantized = quantized.transpose(1, 2)
        
        # Se commit_loss for uma lista (dependendo da versão), some-os
        if isinstance(commit_loss, list) or isinstance(commit_loss, tuple):
             commit_loss = sum(commit_loss)

        return quantized, commit_loss