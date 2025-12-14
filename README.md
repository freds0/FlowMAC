<div align="center">

# 🌊 FlowMAC: Conditional Flow Matching for Audio Coding at Low Bit Rates

### Nicola Pia, Martin Strauss, Markus Multrus, and Bernd Edler

[![python](https://img.shields.io/badge/-Python_3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

> **Abstract:** This is an implementation of **FlowMAC**, a novel neural audio codec for high-quality general audio compression at low bit rates based on Conditional Flow Matching (CFM). FlowMAC jointly learns a mel spectrogram encoder, quantizer (RVQ), and decoder. At inference time, the decoder integrates a continuous normalizing flow via an ODE solver to generate high-quality mel spectrograms from the quantized latent representation.

This codebase is adapted from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) to support **Audio-to-Audio** reconstruction tasks instead of Text-to-Speech.

## Key Features

- **Extreme Compression:** Targets high quality at low bitrates (e.g., 3 kbps).
- **Conditional Flow Matching:** Uses CFM instead of GANs or Diffusion for stable and efficient training.
- **Scalable Architecture:** Based on U-Net and Transformer components.
- **Fast Inference:** Supports Euler ODE solvers with few steps (NFE) for faster-than-real-time generation on CPU.

## Installation

1. Create an environment

```bash
conda create -n flowmac python=3.10 -y
conda activate flowmac

```

2. Install dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install Vector Quantization library (Required for FlowMAC)
pip install vector-quantize-pytorch

```

3. Install the package in editable mode

```bash
pip install -e .

```

##Dataset PreparationFlowMAC is an audio codec, so it requires a dataset of audio files (e.g., LJSpeech, LibriTTS, or music).

1. **Prepare Filelists:**
Create text files listing the paths to your `.wav` files. The format expected by the dataloader is simply the path to the audio file (pipes `|` and additional text are ignored).
Example `data/filelists/train.txt`:
```text
/path/to/dataset/wavs/audio1.wav
/path/to/dataset/wavs/audio2.wav

```


2. **Update Configuration:**
Edit `configs/data/ljspeech_flowmac.yaml` (or create a new one) to point to your filelists and adjust audio parameters (sample rate, n_fft, etc.).
```yaml
train_filelist_path: config/data/filelists/train.txt
valid_filelist_path: config/data/filelists/val.txt
sample_rate: 22050 # or 24000 as in the paper
n_feats: 80        # Mel bands

```



##TrainingTo train the FlowMAC model using the default experiment configuration (LJSpeech example):

```bash
python matcha/train.py experiment=flowmac_ljspeech

```

**Key Hyperparameters:**

* `model.encoder_params.hidden_dim`: Dimension of the latent space (default: 768).
* `model.quantizer_params.codebook_size`: Size of the VQ codebook (default: 256).
* `model.quantizer_params.num_quantizers`: Number of RVQ layers (default: 8).
* `train.monitor`: Monitor `val/loss` for checkpointing.

You can override parameters via CLI:

```bash
# Reduce batch size and set a specific run name
python matcha/train.py experiment=flowmac_ljspeech data.batch_size=16 run_name=my_first_run

```

##Inference / ReconstructionDuring training, the model automatically logs reconstructions to TensorBoard. To run inference manually on audio files (Analysis-Synthesis):

*(Note: CLI for pure audio reconstruction is currently under development. You can use the model directly in Python)*

```python
import torch
from matcha.models.flowmac import FlowMAC
from matcha.utils.audio import mel_spectrogram, load_audio

# Load Model
model = FlowMAC.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Load Audio & Convert to Mel
wav, sr = load_audio("input.wav", sr=22050)
mel = mel_spectrogram(wav, ...) 

# Run Analysis-Synthesis
with torch.no_grad():
    output = model.synthesise(mel.unsqueeze(0), n_timesteps=10)
    
generated_mel = output['mel']
# Use a Vocoder (e.g., BigVGAN/HiFiGAN) to convert generated_mel back to waveform

```

##CitationIf you use this code or the FlowMAC architecture, please cite the original paper:

```text
@article{pia2024flowmac,
  title={FlowMAC: Conditional Flow Matching for Audio Coding at Low Bit Rates},
  author={Pia, Nicola and Strauss, Martin and Multrus, Markus and Edler, Bernd},
  journal={arXiv preprint arXiv:2409.17635},
  year={2024}
}

```

##AcknowledgementsThis code is heavily based on [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS). We also acknowledge the use of:

* [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) for the Residual Vector Quantizer.
* [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) for the project structure.

