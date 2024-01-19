import torch
from data import Data
from torch.utils.data import DataLoader

batch_size = 64

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

data = Data()
training_data, test_data = data.training_data, data.test_data
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)