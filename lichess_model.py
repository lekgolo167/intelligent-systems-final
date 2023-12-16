from peewee import Model, IntegerField, TextField, BlobField, FloatField
from collections import OrderedDict
import numpy as np
from torch import nn, Tensor
from random import randrange
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import Optimizer
import torch
import pytorch_lightning as pl
import torch.nn.functional as F


# Define a Peewee model for chess evaluations
class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  evaluation = FloatField()

  class Meta:
    database = None


# Define an IterableDataset for chess evaluations
class EvaluationDataset(IterableDataset):
  
  def __init__(self, count):
    self.count = count
    
  def __iter__(self):
    return self
  
  def __next__(self) -> dict:
    # Get a random evaluation from the database.
    idx = randrange(self.count)
    return self[idx]
  
  def __len__(self) -> int:
    '''
    Get the total number of evaluations in the dataset.
    '''
    return self.count
  
  def __getitem__(self, idx) -> dict:
    '''
    Get the evaluation details for a specific index.
    '''
    eval = Evaluations.get(Evaluations.id == idx+1)
    bin = np.frombuffer(eval.binary, dtype=np.uint8)
    bin = np.unpackbits(bin, axis=0).astype(np.single) 
    eval.evaluation = max(eval.evaluation, -10)
    eval.evaluation = min(eval.evaluation, 10)
    ev = np.array([eval.evaluation]).astype(np.single) 
    return {'binary':bin, 'eval':ev}  

# Define a PyTorch Lightning model for chess evaluation
class EvaluationModel(pl.LightningModule):
  def __init__(self, count:int, learning_rate:float=1e-3, batch_size:int=1024, layer_count:int=10):
    super().__init__()
    self.count = count
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.save_hyperparameters()
    layers = []

    for i in range(layer_count-1):
      layers.append((f'linear-{i}', nn.Linear(800, 800)))
      layers.append((f'relu-{i}', nn.ReLU()))
    layers.append((f'linear-{layer_count-1}', nn.Linear(800, 1)))

    self.seq = nn.Sequential(OrderedDict(layers))


  def forward(self, x):
    '''
    Define inference actions
    '''
    return self.seq(x)


  def training_step(self, batch:dict, batch_idx:int) -> Tensor:
    '''
    Perform a single training step.

    Args:
        batch (dict): Dictionary containing 'binary' and 'eval' keys.
        batch_idx (int): Index of the batch.
    '''
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    #loss = F.mse_loss(y_hat, y)
    self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
    return loss


  def validation_step(self, batch:dict, batch_idx:int) -> None:
    '''
    Perform a single validation step.

    Args:
        batch (dict): Dictionary containing 'binary' and 'eval' keys.
        batch_idx (int): Index of the batch.
    '''
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    labels_hat = torch.argmax(y_hat, dim=1)
    acc = torch.sum(y == labels_hat).item() / (len(y)*1.0)
    self.log_dict({'val_loss':loss, 'val_acc':acc})

#   def on_train_epoch_end(self) -> None:
#     loss = sum(output['loss'] for output in outputs) / len(outputs)
#     print(loss)


  def configure_optimizers(self) -> Optimizer:
    ''''
      Configure the optimizer for training.
      '''
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


  def train_dataloader(self) -> DataLoader:
    '''
    Create a DataLoader for the training dataset.
    '''
    dataset = EvaluationDataset(self.count)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True)
