import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from core.model import ClipHateMemeModelFreeze, PretrainedModel
from core.dataset import ClipHatefulMemeDatasetFreeze
from core.loop import train_freeze, evaluate_freeze
from torchsummary import summary
from torch.nn import functional as F
import numpy as np
# from madgrad import MADGRAD
from torch.utils.data import WeightedRandomSampler
import json

cur_dir = os.path.dirname(__file__)
DATA_DIR = os.path.join(cur_dir, 'data/')

from core.dataset import HatefulMemesDataset



def main():
	lr = 1e-4
	epochs = 20
	batch_size = 48
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




	train_dataset = ClipHatefulMemeDatasetFreeze(os.path.join(DATA_DIR, 'train.jsonl'), random=True)
	weights = 1. / np.array([5481, 3019])
	sample_weights = torch.tensor([weights[el['label']] for el in train_dataset.data])

	sampler = WeightedRandomSampler(sample_weights, 8500, replacement=True)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

	val_dataset = ClipHatefulMemeDatasetFreeze(os.path.join(DATA_DIR, 'dev.jsonl'), random=False)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
      
	test_dataset = ClipHatefulMemeDatasetFreeze(os.path.join(DATA_DIR, 'test_seen.jsonl'), random=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	model = ClipHateMemeModelFreeze()
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


	criterion = nn.CrossEntropyLoss()#label_smoothing=0.1)
    
	model.to(device)
	criterion.to(device)
	summary(model, input_size=[(1, 768), (1, 1024)])

	train_freeze(
	model, train_loader, val_loader, optimizer, criterion, epochs, device,
	os.path.join(cur_dir, 'models/')
	)
      
	evaluate_freeze(model, test_loader, criterion, device)


if __name__ == '__main__':
    main()
