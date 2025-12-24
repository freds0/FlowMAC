import pyrootutils
import random

# Setup project root
# search_from=__file__ garante que ele encontre a raiz (.project-root) 
# subindo a partir da pasta 'matcha/'
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import os

# Project import
from flowmac.utils.audio import mel_spectrogram

class AudioDataset(Dataset):
    """
    Dataset class to handle audio loading in parallel workers.
    """
    def __init__(self, filepaths, sr, root_path, data_dir=None):
        self.filepaths = filepaths
        self.sr = sr
        self.root = root_path
        self.data_dir = data_dir

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        
        # Path resolution logic
        if not os.path.exists(filepath):
            filepath_alt = os.path.join(self.root, filepath)
            if os.path.exists(filepath_alt):
                filepath = filepath_alt
            elif self.data_dir:
                 filepath = os.path.join(self.data_dir, filepath)

        try:
            # Load and resample
            audio, loaded_sr = torchaudio.load(filepath)
            
            # Mono mix
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if loaded_sr != self.sr:
                resampler = torchaudio.transforms.Resample(loaded_sr, self.sr)
                audio = resampler(audio)
                
            return audio
        except Exception as e:
            # Return None to handle errors in collate_fn
            return None

def collate_fn_no_pad(batch):
    """
    Custom collate function to filter out failed loads (None) 
    and return a list of tensors instead of a padded batch.
    This avoids adding zeros that would skew the statistics.
    """
    return [item for item in batch if item is not None]

def load_filelist(filelist_path):
    if not os.path.isabs(filelist_path):
        filelist_path = os.path.join(root, filelist_path)
    
    print(f"Reading file list at: {filelist_path}")
    with open(filelist_path, 'r', encoding='utf-8') as f:
        filepaths = [line.split('|')[0].strip() for line in f]
    return filepaths

# --- ALTERAÇÃO PRINCIPAL AQUI ---
# Mudamos config_path="configs" para "../configs"
# Isso permite que o Hydra encontre a pasta configs na raiz do projeto
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    
    print("\n" + "="*50)
    print("   COMPUTING DATASET STATISTICS (OPTIMIZED)")
    print("="*50)

    data_cfg = cfg.data
    
    # Parameters
    sample_size = cfg.get("sample_size", None)
    filelist_path = data_cfg.train_filelist_path
    sr = data_cfg.sample_rate
    n_fft = data_cfg.n_fft
    n_mels = data_cfg.n_feats
    hop_length = data_cfg.hop_length
    win_length = data_cfg.win_length
    f_min = data_cfg.f_min
    f_max = data_cfg.f_max
    
    # Configuration for DataLoader
    # Use 80% of available cores or at least 4
    num_workers = max(4, int(os.cpu_count() * 0.8))
    batch_size = 32 # Number of files to load in parallel chunks
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Config:")
    print(f" - Sample Rate: {sr}")
    print(f" - Workers: {num_workers}")
    print(f" - Batch Size: {batch_size}")
    print(f" - Device: {device}")

    # 1. Prepare Dataset and DataLoader
    filepaths = load_filelist(filelist_path)

    if sample_size is not None:
            sample_size = int(sample_size)
            if sample_size < len(filepaths):
                print(f"\n⚠️  ACTIVE SAMPLING: Using {sample_size} random files from a total of {len(filepaths)}.")
                # Shuffle to ensure the sample is representative (variety of speakers/phrases)
                random.shuffle(filepaths)
                # Cut the list
                filepaths = filepaths[:sample_size]

    data_dir = cfg.paths.get("data_dir", None)
    
    dataset = AudioDataset(filepaths, sr, root, data_dir)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn_no_pad,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Accumulators (using float64 for precision)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_elements = 0

    print("Processing audio files...")
    
    # Disable gradient calculation for speed
    with torch.no_grad():
        for batch in tqdm(loader):
            # batch is a list of tensors of different lengths
            for audio in batch:
                audio = audio.to(device)
                
                # Generate Mel
                mel = mel_spectrogram(
                    audio, 
                    n_fft=n_fft, 
                    num_mels=n_mels, 
                    sampling_rate=sr, 
                    hop_size=hop_length, 
                    win_size=win_length, 
                    fmin=f_min, 
                    fmax=f_max, 
                    center=False
                )
                
                # Accumulate
                # Optimization: Convert to float once per tensor to reduce overhead
                total_sum += torch.sum(mel).item()
                total_sq_sum += torch.sum(mel ** 2).item()
                total_elements += mel.numel()

    if total_elements == 0:
        print("ERROR: No files were processed successfully.")
        return

    # Results
    mel_mean = total_sum / total_elements
    mel_variance = (total_sq_sum / total_elements) - (mel_mean ** 2)
    mel_std = math.sqrt(mel_variance)

    print("\n" + "="*50)
    print("   RESULTS")
    print("="*50)
    print(f"data_statistics:")
    print(f"  mel_mean: {mel_mean:.6f}")
    print(f"  mel_std: {mel_std:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
