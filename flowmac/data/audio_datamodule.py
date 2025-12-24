import random
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from flowmac.utils.audio import mel_spectrogram
from flowmac.utils.model import normalize, fix_len_compatibility

class AudioDataset(Dataset):
    def __init__(self, filelist_path, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max, mel_mean, mel_std):
        # Reads audio file list
        with open(filelist_path, encoding="utf-8") as f:
            self.filepaths = [line.strip().split('|')[0] for line in f]
            
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        audio, sr = ta.load(filepath)
        
        # Resample if necessary (The paper uses 24kHz)
        if sr != self.sample_rate:
            audio = ta.functional.resample(audio, sr, self.sample_rate)
            
        # Generate Mel Spectrogram
        mel = mel_spectrogram(
            audio, self.n_fft, self.n_mels, self.sample_rate, 
            self.hop_length, self.win_length, self.f_min, self.f_max, center=False
        ).squeeze()
        
        # Normalization using stats from YAML
        mel = normalize(mel, self.mel_mean, self.mel_std) 
        
        return {"y": mel, "filepath": filepath}

    def __len__(self):
        return len(self.filepaths)

class AudioBatchCollate:
    def __call__(self, batch):
        # Finds the maximum length in the batch and adjusts for reduction compatibility
        max_len = max([x["y"].shape[-1] for x in batch])
        max_len = fix_len_compatibility(max_len)
        
        n_feats = batch[0]["y"].shape[0]
        B = len(batch)
        
        y_padded = torch.zeros((B, n_feats, max_len), dtype=torch.float32)
        y_lengths = []
        
        for i, item in enumerate(batch):
            y = item["y"]
            length = y.shape[-1]
            y_lengths.append(length)
            y_padded[i, :, :length] = y
            
        return {
            "y": y_padded, 
            "y_lengths": torch.tensor(y_lengths, dtype=torch.long)
        }

class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed=None,
    ):
        super().__init__()
        # Saves all arguments to self.hparams
        self.save_hyperparameters(logger=False)
        
    def setup(self, stage=None):
        # Load stats from config
        mel_mean = self.hparams.data_statistics['mel_mean']
        mel_std = self.hparams.data_statistics['mel_std']

        self.trainset = AudioDataset(
            self.hparams.train_filelist_path, 
            self.hparams.n_fft, 
            self.hparams.n_feats,
            self.hparams.sample_rate, 
            self.hparams.hop_length, 
            self.hparams.win_length,
            self.hparams.f_min, 
            self.hparams.f_max,
            mel_mean, 
            mel_std
        )
        self.validset = AudioDataset(
            self.hparams.valid_filelist_path, 
            self.hparams.n_fft, 
            self.hparams.n_feats,
            self.hparams.sample_rate, 
            self.hparams.hop_length, 
            self.hparams.win_length,
            self.hparams.f_min, 
            self.hparams.f_max,
            mel_mean, 
            mel_std
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, # Now using the correct parameter
            collate_fn=AudioBatchCollate()
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            collate_fn=AudioBatchCollate()
        )