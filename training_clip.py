import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from core.model import PretrainedModel, ClipHateMemeModel
from core.dataset import ClipHatefulMemesDataset, collate_fn_clip
from core.loop import train, evaluate

DATA_DIR = '/Users/nstanishev/Workspace/epfl/04/dl/project/data/hateful_memes'


def main():
	lr = 1e-5
	epochs = 10
	batch_size = 64
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model, transform = PretrainedModel.load_clip_model()
	tokenizer = PretrainedModel.load_clip_tokenizer()

	train_dataset = ClipHatefulMemesDataset(
	os.path.join(DATA_DIR, 'train.jsonl'),
	transform=transform,
		tokenizer=tokenizer
	)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_clip)
	val_dataset = ClipHatefulMemesDataset(
	os.path.join(DATA_DIR, 'dev.jsonl'),
	transform=transform,
		tokenizer=tokenizer
	)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_clip)
	test_set = ClipHatefulMemesDataset(
	os.path.join(DATA_DIR, 'test_seen.jsonl'),
	transform=transform,
		tokenizer=tokenizer
	)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_clip)

	model = ClipHateMemeModel(model, tokenizer)
	optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-4)
	criterion = nn.CrossEntropyLoss()

	model.to(device)
	criterion.to(device)

	train(
	model, train_loader, val_loader, optimizer, criterion, epochs, device,
	'/content/models/'
	)
	evaluate(model, test_loader, criterion, device)


if __name__ == '__main__':
    main()
