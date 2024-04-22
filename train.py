import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import nn, optim
import numpy as np

from core.model import PretrainedModel, HateMemeModel
from core.dataset import HatefulMemesDataset, collate_fn
from core.loop import train, evaluate

DATA_DIR = '/home/nstanishev/dl/data/hateful_memes'

def main():
    lr = 0.000005
    epochs = 15
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((232, 232)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tokenizer = PretrainedModel.load_bert_tokenizer()

    train_dataset = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'train.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    weights = 1. / np.array([5481, 3019])
    sample_weights = torch.tensor([weights[el['label']] for el in train_dataset.data])
    sampler = WeightedRandomSampler(sample_weights, 2 * 3019, replacement=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # val_dataset = HatefulMemesDataset(
    #     os.path.join(DATA_DIR, 'dev.jsonl'),
    #     tokenizer=tokenizer,
    #     transform=transform
    # )
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
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_seen_loader = DataLoader(val_seen_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # test_set = HatefulMemesDataset(
    #     os.path.join(DATA_DIR, 'test.jsonl'),
    #     tokenizer=tokenizer,
    #     transform=transform
    # )
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
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_seen_loader = DataLoader(test_seen_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_unseen_loader = DataLoader(test_unseen_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = HateMemeModel(PretrainedModel.load_bert_text_model(), PretrainedModel.load_resnet_image_model(), 0.4)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    train(
        model, train_loader, train_eval_loader, val_seen_loader, val_unseen_loader, optimizer, criterion, epochs, device,
        '/home/nstanishev/dl/models/01'
    )
    evaluate(model, test_seen_loader, criterion, device, 'Test Seen')
    evaluate(model, test_unseen_loader, criterion, device, 'Test Unseen')


if __name__ == '__main__':
    main()
