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

cur_dir = os.path.dirname(__file__)
DATA_DIR = os.path.join(cur_dir, 'data/')



def main():
	lr = 5e-4
	epochs = 20
	batch_size = 64
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	train_dataset = ClipHatefulMemeDatasetFreeze(os.path.join(DATA_DIR, 'train_clip.jsonl'))
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	val_dataset = ClipHatefulMemeDatasetFreeze(os.path.join(DATA_DIR, 'eval_clip.jsonl'))
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

	model = ClipHateMemeModelFreeze()
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
	criterion = nn.CrossEntropyLoss(weight=torch.tensor([1000/5481, 1000/3019]), label_smoothing=0.2)
	#criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=torch.tensor([1.0/5481, 1/3019]))#, pos_weight=torch.ones([1,1]))#, label_smoothing=0.01)

	model.to(device)
	criterion.to(device)
	print(summary(model, input_size=[(1, 768), (1, 768)]))

	train_freeze(
	model, train_loader, val_loader, optimizer, criterion, epochs, device,
	os.path.join(cur_dir, 'models/')
	)


if __name__ == '__main__':
    main()
