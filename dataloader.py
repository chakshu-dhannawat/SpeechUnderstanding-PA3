import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

    def __len__(self):
        total_samples = 0
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            total_samples += len(os.listdir(class_path))
        return total_samples

    def __getitem__(self, idx):
        class_idx = 0
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            num_samples = len(os.listdir(class_path))
            if idx < num_samples:
                file_name = os.listdir(class_path)[idx]
                audio_path = os.path.join(class_path, file_name)
                waveform, sample_rate = torchaudio.load(audio_path)
                if self.transform:
                    waveform = self.transform(waveform)
                return waveform, class_idx
            else:
                idx -= num_samples
                class_idx += 1

def get_dataloader(root_dir, batch_size=32, shuffle=True, transform=None):
    dataset = AudioDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
