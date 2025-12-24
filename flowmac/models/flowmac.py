# matcha/models/flowmac.py

import itertools
import torch
import torch.nn.functional as F
from flowmac.models.baselightningmodule import BaseLightningClass
from flowmac.models.components.flow_matching import CFM

# Import custom components
# Assuming MelEncoder was defined as per previous instructions
# Assuming QuantizerWrapper was defined in components/quantizer.py
from flowmac.models.components.quantizer import QuantizerWrapper

# We reuse the Encoder class because the paper states: 
# "The decoder architecture follows the same structure as the encoder"
from flowmac.models.components.mel_encoder import MelEncoder 

from flowmac.utils.utils import plot_tensor

class FlowMAC(BaseLightningClass): 
    def __init__(
        self,
        n_feats,
        encoder_params,
        decoder_params,
        cfm_params,
        quantizer_params,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.n_feats = n_feats

        # 1. Mel Encoder
        # Extracts latent features from audio
        self.encoder = MelEncoder(
            n_feats=n_feats, 
            hidden_dim=encoder_params['hidden_dim'],
            n_layers=encoder_params['n_layers'],
            n_heads=encoder_params['n_heads']
        )

        # 2. Quantizer (RVQ)
        # Compresses the latent to the target bitrate
        self.quantizer = QuantizerWrapper(
            dim=encoder_params['hidden_dim'],
            codebook_size=quantizer_params['codebook_size'], # 256
            num_quantizers=quantizer_params['num_quantizers'] # 8
        )

        # 3. Mel Decoder (Auxiliary)
        # Necessary to reconstruct the "coarse" Mel used as a condition for the CFM
        # and to calculate the reconstruction loss (L_prior).
        self.mel_decoder = MelEncoder( # Uses similar architecture to the encoder
            n_feats=encoder_params['hidden_dim'], # Input is the latent
            hidden_dim=encoder_params['hidden_dim'],
            n_layers=encoder_params['n_layers'],
            n_heads=encoder_params['n_heads']
        )
        # Final projection to return to Mel dimension (cited in section III-A of the paper)
        self.post_quant_proj = torch.nn.Conv1d(encoder_params['hidden_dim'], n_feats, 1)

        # 4. The CFM Decoder
        # CFM receives: [Noise (n_feats) + Condition (n_feats)] via concatenation.
        # Therefore, in_channels of the internal estimator must be 2 * n_feats.
        self.decoder = CFM(
            in_channels=2 * n_feats, 
            out_channel=n_feats,
            cfm_params=cfm_params,
            decoder_params=decoder_params,
            n_spks=1, 
            spk_emb_dim=0,
        )

    def forward(self, y, y_lengths, spks=None):
        """
        FlowMAC training forward pass
        Args:
            y: Target Mel-spectrogram [Batch, n_feats, Time]
            y_lengths: Real lengths [Batch]
        """
        
        # Create mask
        y_mask = torch.unsqueeze(
            torch.arange(y.size(2), device=y.device) < y_lengths.unsqueeze(1), 1
        ).float()

        # --- 1. Encode ---
        # z: [Batch, hidden_dim, Time]
        z = self.encoder(y, y_mask)

        # --- 2. Quantize ---
        # z_q: [Batch, hidden_dim, Time]
        z_q, loss_q = self.quantizer(z)

        loss_q = loss_q.mean()

        # --- 3. Mel Decoding (Auxiliary Reconstruction) ---
        # Decodes z_q to get a Mel approximation
        # This serves to calculate L_prior and condition the CFM
        decoded_latent = self.mel_decoder(z_q, y_mask)
        y_hat_aux = self.post_quant_proj(decoded_latent) # [Batch, n_feats, Time]

        # --- 4. Flow Matching ---
        # CFM tries to generate 'y' (target) conditioned on 'y_hat_aux'
        # Note: Matcha's compute_loss method generally expects 'mu' as the condition.
        loss_cfm, _ = self.decoder.compute_loss(
            x1=y,          # Target (Real Mel)
            mask=y_mask, 
            mu=y_hat_aux,  # Condition (Decoded Mel from VQ)
            spks=None
        )

        # --- 5. Loss Calculation ---
        # Weights defined in Equation (3) of the paper
        lambda_p = 0.01
        lambda_v = 0.25

        # L_prior: Mel Spectrogram Reconstruction (MSE + MAE)
        loss_mse = F.mse_loss(y_hat_aux, y, reduction='none')
        loss_mae = F.l1_loss(y_hat_aux, y, reduction='none')
        loss_prior = torch.sum((loss_mse + loss_mae) * y_mask) / (torch.sum(y_mask) * self.n_feats)

        # Total Loss
        loss_total = (lambda_p * loss_prior) + (lambda_v * loss_q) + loss_cfm

        return loss_total, loss_cfm, loss_q, loss_prior

    @torch.inference_mode()
    def synthesise(self, y, n_timesteps, temperature=1.0, length_scale=1.0):
        # For inference (Analysis-Synthesis)
        
        # Simple mask for inference
        mask = torch.ones_like(y[:, :1, :])

        # 1. Encode
        z = self.encoder(y, mask)
        
        # 2. Quantize
        z_q, _ = self.quantizer(z)
        
        # 3. Auxiliary Decode (Get the condition)
        decoded_latent = self.mel_decoder(z_q, mask)
        y_hat_aux = self.post_quant_proj(decoded_latent)

        # 4. CFM Sampling
        # Generates high-fidelity Mel from noise + condition (y_hat_aux)
        generated_mel = self.decoder(
            mu=y_hat_aux, 
            mask=mask, 
            n_timesteps=n_timesteps, 
            temperature=temperature
        )
        
        return {
            "mel": generated_mel,
            "mel_aux": y_hat_aux, # Also returns the "low quality" version from VQ
            "z_q": z_q
        }
    
    def training_step(self, batch, batch_idx):
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch.get("spks", None)
        
        loss_total, loss_cfm, loss_q, loss_prior = self(y, y_lengths, spks)
        
        self.log("loss/train", loss_total, prog_bar=True, logger=True, sync_dist=True)
        self.log("sub_loss/train_cfm", loss_cfm, logger=True, sync_dist=True)
        self.log("sub_loss/train_quant", loss_q, logger=True, sync_dist=True)
        self.log("sub_loss/train_prior", loss_prior, logger=True, sync_dist=True)
        
        return loss_total

    def validation_step(self, batch, batch_idx):
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch.get("spks", None)
        
        loss_total, loss_cfm, loss_q, loss_prior = self(y, y_lengths, spks)
        
        self.log("loss/val", loss_total, prog_bar=True, logger=True, sync_dist=True)
        self.log("sub_loss/val_cfm", loss_cfm, logger=True, sync_dist=True)
        self.log("sub_loss/val_quant", loss_q, logger=True, sync_dist=True)
        self.log("sub_loss/val_prior", loss_prior, logger=True, sync_dist=True)
        
        return loss_total

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            # Import wandb only if necessary to avoid errors
            try:
                import wandb
            except ImportError:
                wandb = None

            # Get a batch for visualization
            one_batch = next(iter(self.trainer.val_dataloaders))
            
            # Plot originals in the first epoch
            if self.current_epoch == 0:
                for i in range(min(2, one_batch["y"].shape[0])):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    numpy_img = plot_tensor(y.squeeze().cpu())
                    
                    # === Hybrid WandB / TensorBoard Logic ===
                    if hasattr(self.logger.experiment, "add_image"): # TensorBoard
                        self.logger.experiment.add_image(
                            f"original/{i}",
                            numpy_img,
                            self.current_epoch,
                            dataformats="HWC",
                        )
                    elif wandb is not None and hasattr(self.logger.experiment, "log"): # WandB
                        self.logger.experiment.log(
                            {f"original/{i}": [wandb.Image(numpy_img, caption=f"Original {i}")]},
                        )

            # Synthesize (Reconstruction)
            for i in range(min(2, one_batch["y"].shape[0])):
                y = one_batch["y"][i].unsqueeze(0).to(self.device)
                
                # Run inference
                output = self.synthesise(y, n_timesteps=10)
                
                # Generate numpy arrays of images
                img_cfm = plot_tensor(output["mel"].squeeze().cpu())
                img_aux = plot_tensor(output["mel_aux"].squeeze().cpu())
                
                # === Hybrid WandB / TensorBoard Logic ===
                if hasattr(self.logger.experiment, "add_image"): # TensorBoard
                    self.logger.experiment.add_image(
                        f"generated_cfm/{i}",
                        img_cfm,
                        self.current_epoch,
                        dataformats="HWC",
                    )
                    self.logger.experiment.add_image(
                        f"generated_aux/{i}",
                        img_aux,
                        self.current_epoch,
                        dataformats="HWC",
                    )
                elif wandb is not None and hasattr(self.logger.experiment, "log"): # WandB
                    self.logger.experiment.log(
                        {
                            f"generated_cfm/{i}": [wandb.Image(img_cfm, caption=f"CFM Epoch {self.current_epoch}")],
                            f"generated_aux/{i}": [wandb.Image(img_aux, caption=f"Aux Epoch {self.current_epoch}")]
                        },
                    )