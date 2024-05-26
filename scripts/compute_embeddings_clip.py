import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from core.model import PretrainedModel
import clip
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
import numpy as np
from core.augmentation import augment_image, paraphrase_text

def makeClipDataset(new_data_file_path, source_data_path, model_name):
	if not os.path.exists(source_data_path):
		print('file not found:', source_data_path)
		return
	if os.path.exists(new_data_file_path):
		print('file already exists, remove it to proceed:', new_data_file_path)
		return
	data_dir = os.path.dirname(source_data_path)
	model, transform = PretrainedModel.load_clip_model(model_name)
	tokenizer = PretrainedModel.load_clip_tokenizer()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device=device)
	data = [json.loads(jline) for jline in open(source_data_path, 'r').readlines()]

	with open(new_data_file_path, 'w') as f:
		for line in tqdm(data, total=len(data)):
			text = line['text']
			text = tokenizer(text, context_length=77, truncate=True).to(device=device)
			text = model.encode_text(text).to(dtype=torch.float32)
			img = Image.open(os.path.join(data_dir, line['img']))
			img = transform(img).unsqueeze(0).to(device=device)
			img = model.encode_image(img).to(dtype=torch.float32)
			f.write(json.dumps({'img_embedding': [img.cpu().detach().numpy().tolist()], 'text_embedding': [text.cpu().detach().numpy().tolist()], 'label': line['label']}) + '\n')

def makeClipAugmentedDataset(new_data_file_path, source_data_path, model_name):
	if not os.path.exists(source_data_path):
		print('file not found:', source_data_path)
		return
	if os.path.exists(new_data_file_path):
		print('file already exists, remove it to proceed:', new_data_file_path)
		return
	data_dir = os.path.dirname(source_data_path)

	clip_model, clip_transform = PretrainedModel.load_clip_model(model_name)
	clip_tokenizer = PretrainedModel.load_clip_tokenizer()
	paraphraser_model = PretrainedModel.load_para_t5_model()
	paraphrase_tokenizer = PretrainedModel.load_t5_tokenizer()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	clip_model = clip_model.to(device=device)
	data = [json.loads(jline) for jline in open(source_data_path, 'r').readlines()]

	with open(new_data_file_path, 'w') as f:
		for line in tqdm(data, total=len(data)):
			print('#', end='')
			text = line['text']
			paraphrase = [text]
			paraphrase.extend(paraphrase_text(text, model=paraphraser_model, tokenizer=paraphrase_tokenizer, device=device, return_number=10))
			text_embeddings = []
			for t in paraphrase:
				text = clip_tokenizer(t, context_length=77, truncate=True).to(device=device)
				text_embeddings.append(clip_model.encode_text(text).to(dtype=torch.float32).cpu().detach().numpy().tolist())
			
			img = Image.open(os.path.join(data_dir, line['img']))
			img_embeddings = []
			img_embeddings += [clip_model.encode_image(clip_transform(img).unsqueeze(0).to(device=device)).to(dtype=torch.float32).cpu().detach().numpy().tolist()]
			for i in range(10):
				img_aug = augment_image(img)
				img_aug = clip_transform(img_aug).unsqueeze(0).to(device=device)
				img_embeddings.append(clip_model.encode_image(img_aug).to(dtype=torch.float32).cpu().detach().numpy().tolist())
			f.write(json.dumps({'img_embedding': img_embeddings, 'text_embedding': text_embeddings, 'label': line['label']}) + '\n')


