
import PIL
import torch 
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import torchvision.transforms as T
import os
from model import PretrainedModel

#grayscale
grayscale_transform = T.Grayscale(3)
grayscale = [grayscale_transform]

#random rotation
random_rotation_transformation_45 = T.RandomRotation(45)
# random_rotation_transformation_85 = T.RandomRotation(85)
random_rotation_transformation_65 = T.RandomRotation(65)
random_flip_transform = T.RandomVerticalFlip()
rotations = [random_rotation_transformation_45,random_rotation_transformation_65, random_flip_transform]

#Gausian Blur
gausian_blur_transformation_13 = T.GaussianBlur(kernel_size = (7,13), sigma = (6 , 9))
gausian_blur_transformation_56 = T.GaussianBlur(kernel_size = (7,13), sigma = (5 , 8))
blurs = [gausian_blur_transformation_13,gausian_blur_transformation_56]

#Gausian Noise
def addnoise(input_image, noise_factor = 0.3):
    inputs = T.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip (noisy,0,1.)
    output_image = T.ToPILImage()
    image = output_image(noisy)
    return image

#Colour Jitter
colour_jitter_transformation_1 = T.ColorJitter(brightness=(0.5,1.5),contrast=(3),saturation=(0.3,1.5),hue=(-0.1,0.1))
colour_jitter_transformation_2 = T.ColorJitter(brightness=(0.7),contrast=(6),saturation=(0.9),hue=(-0.1,0.1))
colour_jitter_transformation_3 = T.ColorJitter(brightness=(0.5,1.5),contrast=(2),saturation=(1.4),hue=(-0.1,0.5))
colour_jitters = [colour_jitter_transformation_1,colour_jitter_transformation_2,colour_jitter_transformation_3]

#Random invert
random_invert_transform = T.RandomInvert()
invert = [random_invert_transform]

#Main function that calls all the above functions to create 11 augmented images from one image

#augmented dataset path
augmented_dataset = "/home/ceru/Documents/2o_semestre_23-24/deep_learning/hate-speach-detection/data/temp_aug"

# master dataset path
master_dataset = "/home/ceru/Documents/2o_semestre_23-24/deep_learning/hate-speach-detection/data/temp_img"

def augment_image(orig_img, save_path=None):
	transformations = [rotations, blurs, colour_jitters]#, invert]
	noise_lvl = np.random.random()
    
	if np.random.random() < 0.5:
		image = addnoise(orig_img, noise_lvl)
    
	for el in transformations:
		if np.random.random() < 0.5:
			t = np.random.choice(el)
			image = t(orig_img)


	if save_path:
		image.save(save_path)

	return image




# text augmentation 

def paraphrase_text(text, model=None, tokenizer=None, device='cuda'):
	if model is None:
		model = PretrainedModel.load_para_t5_model()
	if tokenizer is None:
		tokenizer = PretrainedModel.load_t5_tokenizer()
	
	text =  "paraphrase: " + text + " </s>"
	encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
	input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
	outputs = model.generate(
		input_ids=input_ids, attention_mask=attention_masks,
		max_length=256,
		do_sample=True,
		top_k=120,
		top_p=0.98,
		early_stopping=True,
		num_return_sequences=1
	)
	paraphrases = []
	for output in outputs:
		line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
		paraphrases.append(line)
	return paraphrases[0]

