import os

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from core.model import ClipHateMemeModelFreeze
from core.dataset import ClipHatefulMemeDataset, ClipHatefulMemeDatasetOpenAI, ClipHatefulMemeDatasetPrecomputed
from core.loop import train_clip, evaluate_clip
from torchsummary import summary
from torch.nn import functional as F
import numpy as np
# from madgrad import MADGRAD
from torch.utils.data import WeightedRandomSampler
import argparse
import matplotlib.pyplot as plt

cur_dir = os.path.dirname(__file__)
DATA_DIR = os.path.join(cur_dir, 'data/')

def plot_results(train_results, eval_results, test_results, custom_test_results=None, filename='results.pdf'):
	test_results = [test_results for _ in range(len(train_results))]
	if custom_test_results is not None:
		custom_test_results = [custom_test_results for _ in range(len(train_results))]
	train_loss = [result['loss'] for result in train_results]
	train_macro_f1 = [result['macro_f1'] for result in train_results]
	train_micro_f1 = [result['micro_f1'] for result in train_results]
	train_accuracy = [result['accuracy'] for result in train_results]
	train_precision = [result['precision'] for result in train_results]
	train_recall = [result['recall'] for result in train_results]
	train_roc_auc = [result['ROC-AUC'] for result in train_results]

	eval_loss = [result['loss'] for result in eval_results]
	eval_macro_f1 = [result['macro_f1'] for result in eval_results]
	eval_micro_f1 = [result['micro_f1'] for result in eval_results]
	eval_accuracy = [result['accuracy'] for result in eval_results]
	eval_precision = [result['precision'] for result in eval_results]
	eval_recall = [result['recall'] for result in eval_results]
	eval_roc_auc = [result['ROC-AUC'] for result in eval_results]

	test_loss = [result['loss'] for result in test_results]
	test_macro_f1 = [result['macro_f1'] for result in test_results]
	test_micro_f1 = [result['micro_f1'] for result in test_results]
	test_accuracy = [result['accuracy'] for result in test_results]
	test_precision = [result['precision'] for result in test_results]
	test_recall = [result['recall'] for result in test_results]
	test_roc_auc = [result['ROC-AUC'] for result in test_results]

	if custom_test_results is not None:
		custom_test_loss = [result['loss'] for result in custom_test_results]
		custom_test_macro_f1 = [result['macro_f1'] for result in custom_test_results]
		custom_test_micro_f1 = [result['micro_f1'] for result in custom_test_results]
		custom_test_accuracy = [result['accuracy'] for result in custom_test_results]
		custom_test_precision = [result['precision'] for result in custom_test_results]
		custom_test_recall = [result['recall'] for result in custom_test_results]
		custom_test_roc_auc = [result['ROC-AUC'] for result in custom_test_results]

	epochs = range(1, len(train_results) + 1)

	plt.figure(figsize=(15, 10))

	plt.subplot(3, 3, 1)
	plt.plot(epochs, train_loss, label='Train Loss')
	plt.plot(epochs, eval_loss, label='Eval Loss')
	plt.plot(epochs, test_loss, label='Test Loss')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_loss, label='Custom Test Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and Evaluation Loss')
	plt.legend()

	plt.subplot(3, 3, 2)
	plt.plot(epochs, train_accuracy, label='Train Accuracy')
	plt.plot(epochs, eval_accuracy, label='Eval Accuracy')
	plt.plot(epochs, test_accuracy, label='Test Accuracy')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_accuracy, label='Custom Test Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Training and Evaluation Accuracy')
	plt.legend()

	plt.subplot(3, 3, 3)
	plt.plot(epochs, train_macro_f1, label='Train Macro F1')
	plt.plot(epochs, eval_macro_f1, label='Eval Macro F1')
	plt.plot(epochs, test_macro_f1, label='Test Macro F1')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_macro_f1, label='Custom Test Macro F1')
	plt.xlabel('Epochs')
	plt.ylabel('Macro F1')
	plt.title('Training and Evaluation Macro F1')
	plt.legend()

	plt.subplot(3, 3, 4)
	plt.plot(epochs, train_micro_f1, label='Train Micro F1')
	plt.plot(epochs, eval_micro_f1, label='Eval Micro F1')
	plt.plot(epochs, test_micro_f1, label='Test Micro F1')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_micro_f1, label='Custom Test Micro F1')
	plt.xlabel('Epochs')
	plt.ylabel('Micro F1')
	plt.title('Training and Evaluation Micro F1')
	plt.legend()

	plt.subplot(3, 3, 5)
	plt.plot(epochs, train_precision, label='Train Precision')
	plt.plot(epochs, eval_precision, label='Eval Precision')
	plt.plot(epochs, test_precision, label='Test Precision')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_precision, label='Custom Test Precision')
	plt.xlabel('Epochs')
	plt.ylabel('Precision')
	plt.title('Training and Evaluation Precision')
	plt.legend()

	plt.subplot(3, 3, 6)
	plt.plot(epochs, train_recall, label='Train Recall')
	plt.plot(epochs, eval_recall, label='Eval Recall')
	plt.plot(epochs, test_recall, label='Test Recall')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_recall, label='Custom Test Recall')
	plt.xlabel('Epochs')
	plt.ylabel('Recall')
	plt.title('Training and Evaluation Recall')
	plt.legend()

	plt.subplot(3, 3, 7)
	plt.plot(epochs, train_roc_auc, label='Train ROC-AUC')
	plt.plot(epochs, eval_roc_auc, label='Eval ROC-AUC')
	plt.plot(epochs, test_roc_auc, label='Test ROC-AUC')
	if custom_test_results is not None:
		plt.plot(epochs, custom_test_roc_auc, label='Custom Test ROC-AUC')
	plt.xlabel('Epochs')
	plt.ylabel('ROC-AUC')
	plt.title('Training and Evaluation ROC-AUC')
	plt.legend()

	plt.tight_layout()
	plt.savefig(filename)
	plt.show()


def main(model_name, augment_image, paraphrase, precomputed, balancing, train_data, eval_data, test_data):
	if model_name not in ['ViT-L/14', 'ViT-B/32', 'openai/clip-vit-large-patch14']:
		print('Model not implemented')
		raise NotImplementedError

	# Hyperparameters
	lr = 4e-5
	batch_size = 64
	epochs = 20
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.manual_seed(42)
	np.random.seed(42)

	################################ DATASETS ################################
	# Load different dataset based on model and statistics
	if model_name == 'openai/clip-vit-large-patch14':
		size = [768, 1024]
		train_dataset = ClipHatefulMemeDatasetOpenAI(os.path.join(DATA_DIR, train_data), augment_image=(augment_image=="True"), paraphrase=(paraphrase=="True"))
		val_dataset = ClipHatefulMemeDatasetOpenAI(os.path.join(DATA_DIR, eval_data), augment_image=False, paraphrase=False)
		test_dataset = ClipHatefulMemeDatasetOpenAI(os.path.join(DATA_DIR, test_data), augment_image=False, paraphrase=False)
	elif precomputed == 'True':
		print('Using precomputed embeddings')
		train_dataset = ClipHatefulMemeDatasetPrecomputed(os.path.join(DATA_DIR, train_data), augment_img=(augment_image=="True"), paraphrase=(paraphrase=="True"))
		val_dataset = ClipHatefulMemeDatasetPrecomputed(os.path.join(DATA_DIR, eval_data), augment_img=False, paraphrase=False)
		test_dataset = ClipHatefulMemeDatasetPrecomputed(os.path.join(DATA_DIR, test_data), augment_img=False, paraphrase=False)
	else:
		train_dataset = ClipHatefulMemeDataset(os.path.join(DATA_DIR, train_data), model_name, augment_img=(augment_image=="True"), paraphrase=(paraphrase=="True"))
		val_dataset = ClipHatefulMemeDataset(os.path.join(DATA_DIR, eval_data), augment_img=False, paraphrase=False)
		test_dataset = ClipHatefulMemeDataset(os.path.join(DATA_DIR, test_data), augment_img=False, paraphrase=False)

	################################ DATALOADERS ################################
	count = 0
	for el in train_dataset:
		count += el[2].item()
	print('Positive samples:', count, 'Negative samples:', len(train_dataset)-count)

	if balancing == 'sampler':
		weights = 1. / np.array([len(train_dataset)-count, count])
		sample_weights = torch.tensor([weights[el['label']] for el in train_dataset.data])
		sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
	else:
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


	################################ MODEL ################################
	if model_name == 'ViT-L/14':
		size = [768, 768]
	elif model_name == 'ViT-B/32':
		size = [512, 512]
	elif model_name == 'openai/clip-vit-large-patch14':
		size = [768, 1024]
	model = ClipHateMemeModelFreeze(size)

	print(model)
	################################ TRAINING ################################
	if balancing == 'weight':
		weights = 1. / np.array([count, len(train_dataset)-count])
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
	criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    
	model.to(device)
	criterion.to(device)
	summary(model, input_size=[(1, size[0]), (1, size[1])])

	dir_name = 'models/'+model_name+'/'
	results = train_clip(
	model, train_loader, val_loader, optimizer, criterion, epochs, device,
	os.path.join(cur_dir, dir_name)
	)

    ################################ TESTING ################################
	best_model_idx = np.argmax([result['ROC-AUC'] for result in results[1]])

	model.load_state_dict(torch.load(f'{dir_name}model_freeze_{best_model_idx + 1}.pth'))
	model.eval()
	test_results = evaluate_clip(model, test_loader, criterion, device)
	if model_name=='ViT-L/14':
		CUSTOM_DATA_DIR = os.path.join(cur_dir, 'Real_Life_Data_2024/')
		test_dataset = ClipHatefulMemeDatasetPrecomputed(os.path.join(CUSTOM_DATA_DIR, 'real_life_data_precomputed.jsonl'), augment_img=False, paraphrase=False)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
		custom_test_results = evaluate_clip(model, test_loader, criterion, device, kind='Custom Test')
	else:
		custom_test_results = None

	################################ PLOTTING ################################
	plot_results(results[0], results[1], test_results, custom_test_results, f'{model_name.split('/')[0]}_aug_{augment_image}_para_{paraphrase}_precomputed_{precomputed}_balancing_{balancing}_train_{train_data.split(".")[0]}.pdf')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a schema')
	parser.add_argument('--model_name', metavar='str', required=False, default='ViT-L/14', help='the clip model to use')
	parser.add_argument('--augment_image', metavar='bool', required=False, default="True", help='use data augmentation')
	parser.add_argument('--paraphrase', metavar='bool', required=False, default="True", help='use data augmentation')
	parser.add_argument('--precomputed', metavar='bool', required=False, default="True", help='use precomputed embeddings')
	parser.add_argument('--balancing', metavar='str', required=False, default='sampler', help='sampler - weight - none')
	parser.add_argument('--train_data', metavar='path', required=False, default='train.jsonl', help='dataset to train on')
	parser.add_argument('--eval_data', metavar='path', required=False, default='dev_seen.jsonl', help='dataset to eval on')
	parser.add_argument('--test_data', metavar='path', required=False, default='test_seen.jsonl', help='dataset to test on')
	args = parser.parse_args()
	main(args.model_name, args.augment_image, args.paraphrase, args.precomputed, args.balancing, args.train_data, args.eval_data, args.test_data)

