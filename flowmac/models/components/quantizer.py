import torch.nn as nn
# Ensure you have installed: pip install vector-quantize-pytorch
from vector_quantize_pytorch import ResidualVQ

class QuantizerWrapper(nn.Module):
    def __init__(self, dim, codebook_size=256, num_quantizers=8):
        super().__init__()
        # Configuration based on the FlowMAC paper:
        # "FlowMAC uses a codebook size of 256 and 8 quantizer stages"
        self.rvq = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            kmeans_init=True,   # Helps with training stability
            kmeans_iters=10,
            threshold_ema_dead_code=2  # Helps to revive dead codes
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Dim, Time] coming from the Encoder
        Returns:
            quantized: Quantized tensor [Batch, Dim, Time]
            commit_loss: Quantization loss
        """
        # The vector-quantize-pytorch library expects [Batch, Time, Dim]
        x = x.transpose(1, 2)
        
        # Performs residual quantization
        # indices are the discrete codes (useful for compression/inference)
        quantized, indices, commit_loss = self.rvq(x)
        
        # Returns to original format [Batch, Dim, Time] for the Decoder
        quantized = quantized.transpose(1, 2)
        
        # If commit_loss is a list (depending on the version), sum them up
        if isinstance(commit_loss, list) or isinstance(commit_loss, tuple):
             commit_loss = sum(commit_loss)

        return quantized, commit_loss