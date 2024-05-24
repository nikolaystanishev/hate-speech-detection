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

DATA_DIR = '/home/nstanishev/dl/data/real'

def main():
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((460, 460)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tokenizer = PretrainedModel.load_bert_tokenizer()

    test_set = HatefulMemesDataset(
        os.path.join(DATA_DIR, 'real_life_data.jsonl'),
        tokenizer=tokenizer,
        transform=transform
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = HateMemeModel(PretrainedModel.load_bert_text_model(), PretrainedModel.load_resnet_image_model(), dropout=0.25)
    model.load_state_dict(torch.load('/home/nstanishev/dl/models/37/model_15.pth'))
    model.to(device)
    model.eval()
    evaluate(model, test_loader, None, device, 'Test')


if __name__ == '__main__':
    main()
