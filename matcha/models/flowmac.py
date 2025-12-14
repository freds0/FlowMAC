# matcha/models/flowmac.py

import itertools
import torch
import torch.nn.functional as F
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM

# Importação dos componentes customizados
# Assumindo que MelEncoder foi definido conforme instrução anterior
# Assumindo que QuantizerWrapper foi definido em components/quantizer.py
from matcha.models.components.quantizer import QuantizerWrapper
# Reutilizamos a classe do Encoder pois o paper diz: 
# "The decoder architecture follows the same structure as the encoder"
from matcha.models.components.mel_encoder import MelEncoder 

from matcha.utils.utils import plot_tensor

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
        # Extrai features latentes do áudio
        self.encoder = MelEncoder(
            n_feats=n_feats, 
            hidden_dim=encoder_params['hidden_dim'],
            n_layers=encoder_params['n_layers'],
            n_heads=encoder_params['n_heads']
        )

        # 2. Quantizador (RVQ)
        # Comprime o latente para bitrate alvo
        self.quantizer = QuantizerWrapper(
            dim=encoder_params['hidden_dim'],
            codebook_size=quantizer_params['codebook_size'], # 256
            num_quantizers=quantizer_params['num_quantizers'] # 8
        )

        # 3. Mel Decoder (Auxiliar)
        # Necessário para reconstruir o Mel "grosseiro" usado como condição para o CFM
        # e para calcular a loss de reconstrução (L_prior).
        self.mel_decoder = MelEncoder( # Usa arquitetura similar ao encoder
            n_feats=encoder_params['hidden_dim'], # Entrada é o latente
            hidden_dim=encoder_params['hidden_dim'],
            n_layers=encoder_params['n_layers'],
            n_heads=encoder_params['n_heads']
        )
        # Projeção final para voltar à dimensão de Mel (citado na seção III-A do paper)
        self.post_quant_proj = torch.nn.Conv1d(encoder_params['hidden_dim'], n_feats, 1)

        # 4. O Decoder CFM
        # O CFM recebe: [Ruído (n_feats) + Condição (n_feats)] via concatenação.
        # Portanto, in_channels do estimador interno deve ser 2 * n_feats.
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
            y: Mel-spectrogram alvo [Batch, n_feats, Time]
            y_lengths: Comprimentos reais [Batch]
        """
        
        # Criar máscara
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

        # --- 3. Mel Decoding (Reconstrução Auxiliar) ---
        # Decodifica z_q para obter uma aproximação do Mel
        # Isso serve para calcular L_prior e condicionar o CFM
        decoded_latent = self.mel_decoder(z_q, y_mask)
        y_hat_aux = self.post_quant_proj(decoded_latent) # [Batch, n_feats, Time]

        # --- 4. Flow Matching ---
        # O CFM tenta gerar 'y' (target) condicionado em 'y_hat_aux'
        # Nota: O método compute_loss do Matcha geralmente espera 'mu' como a condição.
        loss_cfm, _ = self.decoder.compute_loss(
            x1=y,          # Alvo (Mel Real)
            mask=y_mask, 
            mu=y_hat_aux,  # Condição (Mel decodificado do VQ)
            spks=None
        )

        # --- 5. Loss Calculation ---
        # Pesos definidos na Equação (3) do paper
        lambda_p = 0.01
        lambda_v = 0.25

        # L_prior: Reconstrução do Mel Spectrograma (MSE + MAE)
        loss_mse = F.mse_loss(y_hat_aux, y, reduction='none')
        loss_mae = F.l1_loss(y_hat_aux, y, reduction='none')
        loss_prior = torch.sum((loss_mse + loss_mae) * y_mask) / (torch.sum(y_mask) * self.n_feats)

        # Loss Total
        loss_total = (lambda_p * loss_prior) + (lambda_v * loss_q) + loss_cfm

        return loss_total, loss_cfm, loss_q, loss_prior

    @torch.inference_mode()
    def synthesise(self, y, n_timesteps, temperature=1.0, length_scale=1.0):
        # Para inferência (Analysis-Synthesis)
        
        # Máscara simples para inferência
        mask = torch.ones_like(y[:, :1, :])

        # 1. Encode
        z = self.encoder(y, mask)
        
        # 2. Quantize
        z_q, _ = self.quantizer(z)
        
        # 3. Auxiliary Decode (Obter a condição)
        decoded_latent = self.mel_decoder(z_q, mask)
        y_hat_aux = self.post_quant_proj(decoded_latent)

        # 4. CFM Sampling
        # Gera o Mel de alta fidelidade a partir do ruído + condição (y_hat_aux)
        generated_mel = self.decoder(
            mu=y_hat_aux, 
            mask=mask, 
            n_timesteps=n_timesteps, 
            temperature=temperature
        )
        
        return {
            "mel": generated_mel,
            "mel_aux": y_hat_aux, # Retorna também a versão "low quality" do VQ
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
            # Pega um batch para visualização
            one_batch = next(iter(self.trainer.val_dataloaders))
            
            # Plota originais na primeira época
            if self.current_epoch == 0:
                for i in range(min(2, one_batch["y"].shape[0])):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            # Sintetiza (Reconstrução)
            for i in range(min(2, one_batch["y"].shape[0])):
                y = one_batch["y"][i].unsqueeze(0).to(self.device)
                
                # Roda inferência
                output = self.synthesise(y, n_timesteps=10)
                
                # Plota resultados
                self.logger.experiment.add_image(
                    f"generated_cfm/{i}",
                    plot_tensor(output["mel"].squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_aux/{i}",
                    plot_tensor(output["mel_aux"].squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )