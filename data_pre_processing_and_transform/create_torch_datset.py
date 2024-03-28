
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TorchDatasetTS(Dataset):

  def __init__(self, sequences):
    #super().__init__()
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)
  
  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]

    return dict(
      sequence= torch.Tensor(sequence.to_numpy()),
      label = torch.tensor(label.to_numpy(), dtype=torch.float32).reshape(1,-1)
    )
  

class TorchDataModule(pl.LightningDataModule):

  def __init__(self, train_sequences, test_sequences, batch_size):

    super().__init__()
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self):
    self.train_datset = TorchDatasetTS(self.train_sequences)  
    self.test_datset = TorchDatasetTS(self.test_sequences)  

  def train_dataloader(self):
    return DataLoader(
      self.train_datset,
      batch_size = self.batch_size,
      shuffle = False,
      num_workers = 4
    )  
  
  def val_dataloader(self):   # TODO make seprate data for validation
    return DataLoader(
      self.test_datset,
      batch_size = 1,
      shuffle = False,
      num_workers = 4
    )  
  
  def test_dataloader(self):
    return DataLoader(
      self.test_datset,
      batch_size = 1,
      shuffle = False,
      num_workers = 4
    )  
