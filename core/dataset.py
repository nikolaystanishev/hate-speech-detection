import os
import json
from PIL import Image
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):

    def __init__(self, data_file_path, tokenizer=None, transform=None):
        self.data = [json.loads(jline) for jline in open(data_file_path, 'r').readlines()]
        self.data_dir = os.path.dirname(data_file_path)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        el = self.data[idx]

        img = Image.open(os.path.join(self.data_dir, el['img']))
        text = el['text']
        label = el['label']

        if self.transform:
            img = self.transform(img)
        if self.tokenizer:
            text = self.tokenizer(text, return_tensors='pt')

        return img, text, label
