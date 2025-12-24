import argparse
import json
import torch
import torchaudio
import numpy as np
import os
import soundfile as sf
import torch.nn.functional as F

# FlowMAC imports
from flowmac.models.flowmac import FlowMAC
from flowmac.utils.audio import mel_spectrogram
# Import fix_len_compatibility to resolve dimension mismatch errors (U-Net)
from flowmac.utils.model import normalize, denormalize, fix_len_compatibility

# HiFiGAN import
try:
    from flowmac.hifigan.models import Generator as HiFiGANGenerator
    from flowmac.hifigan.env import AttrDict
    HAS_LOCAL_HIFIGAN = True
except ImportError:
    HAS_LOCAL_HIFIGAN = False

def load_audio(file_path, target_sr=22050):
    """Loads and prepares the audio file."""
    wav, sr = torchaudio.load(file_path)
    # Mix to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
    return wav

def load_hifigan(checkpoint_path, config_path, device):
    """Loads the local HiFiGAN Vocoder."""
    if not HAS_LOCAL_HIFIGAN:
        raise ImportError("HiFiGAN modules not found in flowmac/hifigan.")
    
    with open(config_path) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    vocoder = HiFiGANGenerator(h).to(device)
    
    # Safe loading (weights_only=False required for some checkpoints)
    state_dict_g = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle 'generator.' prefixes if present
    new_state_dict = {}
    for k, v in state_dict_g.items():
        if 'generator' in k: 
            new_key = k.replace('generator.', '')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    vocoder.load_state_dict(new_state_dict.get('generator', new_state_dict))
    vocoder.eval()
    vocoder.remove_weight_norm()
    print("HiFiGAN Vocoder loaded!")
    return vocoder

def inspect_compression(model, mel_input):
    """Extracts and prints the compression indices (bitstream)."""
    print("\n" + "="*30)
    print("   COMPRESSION ANALYSIS")
    print("="*30)
    
    with torch.no_grad():
        # Simple mask (all ones)
        mask = torch.ones((mel_input.shape[0], 1, mel_input.shape[2])).to(mel_input.device)
        
        # Transpose to the format expected by the Encoder
        # The Encoder expects [Batch, Channels, Time], but the output here
        # is handled internally. We follow the standard flow.
        z = model.encoder(mel_input, mask)
        
        # The Quantizer expects channels in the last dimension: [Batch, Time, Channels]
        z = z.transpose(1, 2)
        
        # Run through RVQ
        vq_out = model.quantizer.rvq(z)
        indices = vq_out[1]
        
        # Transpose indices back for visualization: [Batch, Layers, Time]
        indices = indices.transpose(1, 2)
        
    indices = indices.squeeze().cpu().numpy()
    n_quantizers, n_frames = indices.shape
    
    print(f"Codes Shape: {indices.shape} (Quantizers x Frames)")
    print(f"Example (First 5 frames):")
    print(indices[:, :5])
    
    # Bitrate Estimation
    sr = 22050
    hop = 256
    nbits = 8 # log2(256)
    kbps = (sr / hop) * n_quantizers * nbits / 1000
    print(f"\nEstimated Bitrate: {kbps:.2f} kbps")
    print("="*30 + "\n")
    return indices

def main():
    parser = argparse.ArgumentParser(description="FlowMAC Inference")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/flowmac_ljspeech/last.ckpt", help="FlowMAC Checkpoint (.ckpt)")
    parser.add_argument("--audio", type=str, required=True, help="Input audio (.wav)")
    
    parser.add_argument("--vocoder_ckpt", type=str, default="./checkpoints/hifigan/g_02500000", help="HiFiGAN Checkpoint path")
    # UPDATED: Default path points to flowmac folder
    parser.add_argument("--vocoder_config", type=str, default="./checkpoints/hifigan/config.json", help="HiFiGAN Config JSON")
    
    parser.add_argument("--use_speechbrain", action="store_true", help="Use SpeechBrain vocoder (if local HiFiGAN is missing)")
    parser.add_argument("--steps", type=int, default=10, help="ODE Solver steps (NFE)")
    parser.add_argument("--temp", type=float, default=0.667, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Inference on device: {args.device} ---")

    # 1. Load FlowMAC
    try:
        # Note: weights_only=False is crucial for loading checkpoints with Hydra configs
        model = FlowMAC.load_from_checkpoint(args.checkpoint, map_location=args.device, weights_only=False)
        model.to(args.device)
        model.eval()
        print("FlowMAC loaded.")
    except Exception as e:
        print(f"Error loading FlowMAC: {e}")
        print("Tip: If the error mentions 'weights_only', verify your torch version or the training hack.")
        return

    # 2. Load Vocoder
    vocoder = None
    if args.use_speechbrain:
        print("Loading SpeechBrain Vocoder...")
        from speechbrain.inference.vocoders import HIFIGAN
        vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmp_vocoder", run_opts={"device": args.device})
    elif args.vocoder_ckpt and args.vocoder_config:
        try:
            vocoder = load_hifigan(args.vocoder_ckpt, args.vocoder_config, args.device)
        except Exception as e:
            print(f"Warning: Failed to load local HiFiGAN ({e}).")

    # 3. Prepare Audio
    # These stats should match your config.yaml
    MEL_MEAN = -5.5264
    MEL_STD = 2.0655
    
    # Ensure SR matches model config (usually 22050 or 24000)
    target_sr = 22050 # Adjust if your model uses 24000
    
    wav = load_audio(args.audio, target_sr=target_sr).to(args.device)
    
    mel_orig = mel_spectrogram(
        wav, 
        n_fft=1024, 
        num_mels=80, 
        sampling_rate=target_sr, 
        hop_size=256, 
        win_size=1024, 
        fmin=0, 
        fmax=8000, 
        center=False
    )
    
    # Handle tensor dimensions
    if mel_orig.dim() == 3:
        mel_input = normalize(mel_orig, MEL_MEAN, MEL_STD)
    else:
        mel_input = normalize(mel_orig, MEL_MEAN, MEL_STD).unsqueeze(0)

    # --- DIMENSION CORRECTION (PADDING) ---
    # Calculate the multiple size needed so U-Net doesn't break.
    current_len = mel_input.shape[-1]
    target_len = fix_len_compatibility(current_len)
    
    if target_len != current_len:
        pad_amount = target_len - current_len
        print(f"Adjusting Mel size: {current_len} -> {target_len} (Padding: {pad_amount})")
        mel_input = F.pad(mel_input, (0, pad_amount), mode='replicate')
    # --------------------------------------

    # 4. Inspect Compression
    inspect_compression(model, mel_input)

    # 5. Reconstruction
    print(f"Reconstructing spectrogram ({args.steps} steps)...")
    with torch.no_grad():
        outputs = model.synthesise(mel_input, n_timesteps=args.steps, temperature=args.temp)
        
        mel_recon_norm = outputs['mel']
        
        # Remove extra padding before saving audio (optional, but good for fidelity)
        if target_len != current_len:
             mel_recon_norm = mel_recon_norm[..., :current_len]

        mel_recon_denorm = denormalize(mel_recon_norm, MEL_MEAN, MEL_STD)

    # 6. Vocoding and Saving
    if vocoder is not None:
        print("Converting Mel to Audio (Vocoding)...")
        with torch.no_grad():
            if args.use_speechbrain:
                # Speechbrain expects (batch, time, channels)
                mel_sb = mel_recon_denorm.transpose(1, 2)
                wav_recon = vocoder.decode_batch(mel_sb)
            else:
                # Standard HiFiGAN expects (batch, channels, time)
                wav_recon = vocoder(mel_recon_denorm)
        
        out_path = os.path.join(args.output_dir, f"recon_{os.path.basename(args.audio)}")
        
        wav_numpy = wav_recon.squeeze().cpu().numpy()
        # Ensure correct format for soundfile (Time,)
        if len(wav_numpy.shape) > 1:
            wav_numpy = wav_numpy.flatten()
            
        sf.write(out_path, wav_numpy, target_sr)
        print(f"\nSuccess! Reconstructed audio saved at: {out_path}")

if __name__ == "__main__":
    main()