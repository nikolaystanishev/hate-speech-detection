import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from core.model import PretrainedModel
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import numpy as np
from core.augmentation import augment_image, paraphrase_text

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
            text = self.tokenizer(text.lower(), return_tensors='pt')

        return img, text, label


def collate_fn(batch):
    lens = [len(row[1]['input_ids'][0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    input_ids_tensor = torch.zeros(bsz, max_seq_len).long()
    token_type_ids_tensor = torch.zeros(bsz, max_seq_len).long()
    attention_mask_tensor = torch.zeros(bsz, max_seq_len).long()

    image_tensor = torch.stack([row[0] for row in batch])

    label_tensor = torch.tensor([row[2] for row in batch])

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        input_ids_tensor[i_batch, :length] = input_row[1]['input_ids'][0]
        token_type_ids_tensor[i_batch, :length] = input_row[1]['token_type_ids'][0]
        attention_mask_tensor[i_batch, :length] = input_row[1]['attention_mask'][0]

    return (
        image_tensor,
        {
            'input_ids': input_ids_tensor,
            'token_type_ids': token_type_ids_tensor,
            'attention_mask': attention_mask_tensor
        },
        label_tensor
    )


class ClipHatefulMemeDatasetOpenAI(Dataset):
    def __init__(self, data_file_path, random=True):
        self.data = [json.loads(jline) for jline in open(data_file_path, 'r').readlines()]
        self.data_dir = os.path.dirname(data_file_path)
        self.data_file_path = data_file_path
        self.embedding_dir = os.path.join(os.path.dirname(data_file_path), 'embeddings' + "_openai/clip-vit-large-patch14")
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        if not os.path.exists(os.path.join(self.embedding_dir, os.path.basename(data_file_path).split('.')[0] + 'embeddings.jsonl')):
            self.embedding_data = {}
        else:
            self.embedding_data = json.load(open(os.path.join(self.embedding_dir, os.path.basename(data_file_path).split('.')[0] + 'embeddings.jsonl'), 'r'))
            print(type(self.embedding_data))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device=self.device)
        self.random = random
        self.augment_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.RandomGrayscale(p=0.1,),
                transforms.ColorJitter(),
                transforms.RandomRotation(20),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
            ])
        self.modified = False
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        el = self.data[idx]

        label = torch.tensor(el['label']).to(self.device)

        
        if self.random and np.random.random()<0.5:
            img = Image.open(os.path.join(self.data_dir, el['img']))
            text = el['text']
            img = self.augment_transform(img)
            img = self.image_processor(images=img, return_tensors="pt")['pixel_values'].to(device=self.device)
            img = self.clip.vision_model(pixel_values=img).pooler_output
            text = self.text_processor(el['text'], padding=True, return_tensors="pt", truncation=True).to(device=self.device)
            text = self.clip.text_model(input_ids = text['input_ids'], attention_mask=text['attention_mask']).pooler_output

        elif self.embedding_data.get(str(idx), None) is not None:
            img = torch.tensor(self.embedding_data[str(idx)]['image_original'])
            text = torch.tensor(self.embedding_data[str(idx)]['text_original'])
        else:
            img = Image.open(os.path.join(self.data_dir, el['img']))
            text = el['text']
            img = self.image_processor(images=img, return_tensors="pt")['pixel_values'].to(device=self.device)
            img = self.clip.vision_model(pixel_values=img).pooler_output
            text = self.text_processor(el['text'], padding=True, return_tensors="pt", truncation=True).to(device=self.device)
            text = self.clip.text_model(input_ids = text['input_ids'], attention_mask=text['attention_mask']).pooler_output
            
            img1 = img
            text1 = text
            self.embedding_data[str(idx)] = {'image_original': img1.cpu().detach().numpy().tolist(), 'text_original': text1.cpu().detach().numpy().tolist(), 'label': label.item()}
            self.modified = True
            

        if type(img) != torch.Tensor or type(text) != torch.Tensor or type(label) != torch.Tensor:
            print(type(img), type(text))
            print(img, text)
        return img.to(device=self.device), text.to(device=self.device), label.to(device=self.device)
    
    def save_embeddings(self):
        if self.modified:
            with open(os.path.join(self.embedding_dir, os.path.basename(self.data_file_path).split('.')[0] + 'embeddings.jsonl'), 'w') as f:
                    json.dump(self.embedding_data, f)
            self.modified = False

class ClipHatefulMemeDatasetPrecomputed(Dataset):
    def __init__(self, data_file_path, augment_img=True, paraphrase=True):
        self.data = [json.loads(jline) for jline in open(data_file_path, 'r').readlines()]
        print('loaded embeddings')
        self.data_dir = os.path.dirname(data_file_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augment_img = augment_img
        self.paraphrase = paraphrase

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        el = self.data[idx]
        if self.augment_img and np.random.random()<0.5:
            i = np.random.randint(1, len(el['img_embedding']))
        else:
            i = 0
        if self.paraphrase and np.random.random()<0.5:
            j = np.random.randint(1, len(el['text_embedding']))
        else:
            j = 0
        img = torch.tensor(el['img_embedding'][i]).to(self.device)
        text = torch.tensor(el['text_embedding'][j]).to(self.device)
        label = torch.tensor(el['label']).to(self.device)
        return img, text, label
    
    def save_embeddings(self):
        return

class ClipHatefulMemeDataset(Dataset):
    def __init__(self, data_file_path, model_name="ViT-L/14", augment_img=True, paraphrase=True):
        self.data = [json.loads(jline) for jline in open(data_file_path, 'r').readlines()]
        self.data_dir = os.path.dirname(data_file_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.image_transform = PretrainedModel.load_clip_model(model_name)
        self.tokenizer = PretrainedModel.load_clip_tokenizer()
        self.paraphrase = paraphrase
        self.augment_img = augment_img
        if self.paraphrase:
            self.paraphrase = paraphrase_text
        else:
            self.paraphrase = lambda x: [x]
        if self.augment_img:
            self.augment_image = augment_image
        else:
            self.augment_image = lambda x: x

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        el = self.data[idx]
        img = Image.open(os.path.join(self.data_dir, el['img']))
        text = el['text']
        if self.augment_img and np.random.random()<0.5:
            img = self.augment_image(img)
        img = self.image_transform(img).unsqueeze(0).to(self.device)
        img = self.model.encode_image(img).to(self.device)
        if self.paraphrase and np.random.random()<0.5:
            text = self.paraphrase(text)[0]
        text = self.tokenizer(text).to(self.device)
        text = self.model.encode_text(text).to(self.device)
        label = torch.tensor(el['label']).to(self.device)
        return img, text, label
    
    def save_embeddings(self):
        return