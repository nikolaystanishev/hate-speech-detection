import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from core.model import PretrainedModel, HateMemeModel
from core.dataset import HatefulMemesDataset, collate_fn
from core.loop import train, evaluate

DATA_DIR = '/Users/nstanishev/Workspace/epfl/04/dl/project/data/hateful_memes'


def main():
    lr = 1e-5
    epochs = 10
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((232, 232)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tokenizer = PretrainedModel.load_bert_tokenizer()

    train_dataset = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'train.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataset = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'dev.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_set = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'test.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = HateMemeModel(PretrainedModel.load_bert_text_model(), PretrainedModel.load_resnet_image_model())
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    train(
        model, train_loader, val_loader, optimizer, criterion, epochs, device,
        '/Users/nstanishev/Workspace/epfl/04/dl/project/models/01'
    )
    evaluate(model, test_loader, criterion, device)


if __name__ == '__main__':
    main()
