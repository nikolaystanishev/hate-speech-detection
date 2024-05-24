import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch import nn, optim
import numpy as np

from core.model import PretrainedModel, HateMemeModel
from core.dataset import HatefulMemesDataset, collate_fn
from core.loop import train, evaluate

DATA_DIR = '/home/nstanishev/dl/data/hateful_memes'

def main():
    lr = 0.00001
    epochs = 15
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((460, 460)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    augment_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((460, 460)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tokenizer = PretrainedModel.load_bert_tokenizer()

    train_dataset = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'train.jsonl'),
        tokenizer=tokenizer,
        transform=augment_transform
    )
    val_seen_dataset = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'dev_seen.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    val_unseen_dataset = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'dev_unseen.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    test_seen_set = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'test_seen.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    test_unseen_set = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'test_unseen.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )

    train_labels = [el['label'] for el in train_dataset.data]
    weights = 1. / np.array([len(train_labels) - sum(train_labels), sum(train_labels)])
    sample_weights = torch.tensor([weights[el] for el in train_labels])
    sampler = WeightedRandomSampler(sample_weights, 15000, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)

    val_seen_loader = DataLoader(val_seen_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_seen_loader = DataLoader(test_seen_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_unseen_loader = DataLoader(test_unseen_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = HateMemeModel(PretrainedModel.load_bert_text_model(), PretrainedModel.load_resnet_image_model(), dropout=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.35, 0.65]))

    # lr_schedule = optim.lr_scheduler.MultiplicativeLR(
    #     optimizer, lr_lambda=lambda epoch: 0.1 if epoch in [3] else 1
    # )
    lr_schedule = None

    model.to(device)
    criterion.to(device)

    train(
        model, train_loader, optimizer, criterion, epochs, device, '/home/nstanishev/dl/models/37',
        {'Validation Seen': val_seen_loader, 'Validation Unseen': val_unseen_loader},
        lr_schedule
    )
    evaluate(model, test_seen_loader, criterion, device, 'Test Seen')
    evaluate(model, test_unseen_loader, criterion, device, 'Test Unseen')


if __name__ == '__main__':
    main()
