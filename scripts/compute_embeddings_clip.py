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

def makeClipDataset(new_data_file_path, old_data_file_path):
	if not os.path.exists(old_data_file_path):
		print('file not found:', old_data_file_path)
		return
	if os.path.exists(new_data_file_path):
		print('file already exists, remove it to procees:', new_data_file_path)
		return
	data_dir = os.path.dirname(old_data_file_path)
	model, transform = PretrainedModel.load_clip_model()
	tokenizer = PretrainedModel.load_clip_tokenizer()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device=device)
	data = [json.loads(jline) for jline in open(old_data_file_path, 'r').readlines()]

	with open(new_data_file_path, 'w') as f:
		for line in tqdm(data, total=len(data)):
			text = line['text']
			text = tokenizer(text, context_length=77, truncate=True).to(device=device)
			text = model.encode_text(text).to(dtype=torch.float32)
			img = Image.open(os.path.join(data_dir, line['img']))
			img = transform(img).unsqueeze(0).to(device=device)
			img = model.encode_image(img).to(dtype=torch.float32)
			f.write(json.dumps({'img_embedding': img.detach().numpy().tolist(), 'text_embedding': text.detach().numpy().tolist(), 'label': line['label']}) + '\n')

makeClipDataset('data/train_clip.jsonl', 'data/train.jsonl')
makeClipDataset('data/eval_clip.jsonl', 'data/dev.jsonl')
makeClipDataset('data/test_clip.jsonl', 'data/test.jsonl')

