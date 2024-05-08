import torch
from torch.nn import Module, Sequential, Linear
from transformers import BertModel, BertTokenizer, AutoModel
import torchvision.models as models
import os
import sys
import json
import clip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HateMemeModel(Module):

    def __init__(self, text_model, image_model, dropout=0.1):
        super(HateMemeModel, self).__init__()
        self.text_model = text_model
        # self.batch_norm1 = torch.nn.BatchNorm1d(256)
        self.image_model = image_model
        # self.batch_norm2 = torch.nn.BatchNorm1d(512)
        
        # 2816
        self.fc1 = Linear(512 + 512, 512)
        self.fc2 = Linear(512, 128)
        self.fc3 = Linear(128, 2)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text, image):
        text_features = self.relu(self.text_model(**text).pooler_output)
        # text_features = self.relu(self.batch_norm1(self.text_model(**text).pooler_output))
        image_features = self.image_model(image)
        image_features = self.relu(image_features.view(image_features.size(0), -1))
        # image_features = self.relu(self.batch_norm2(image_features.view(image_features.size(0), -1)))
        
        combined = torch.cat([text_features, image_features], dim=1)
        out = self.relu(self.fc1(combined))
        # out = self.dropout(self.relu(self.fc1(combined)))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class PretrainedModel:

    @staticmethod
    def load_bert_text_model():
        return AutoModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    
    @staticmethod
    def load_bert_tokenizer():
        return BertTokenizer.from_pretrained('bert-base-uncased')
    
    @staticmethod
    def load_resnet_image_model():
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(model.children())[:-1]
        return Sequential(*modules)

    @staticmethod
    def load_clip_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return clip.load("ViT-L/14", device=device)

    @staticmethod
    def load_clip_tokenizer():
        return clip.tokenize
    
    @staticmethod
    def load_para_t5_model():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)
    
    @staticmethod
    def load_t5_tokenizer():
        return AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  



class ClipHateMemeModelFreeze(Module):
        def __init__(self):
            super(ClipHateMemeModelFreeze, self).__init__()
            self.dropout0 = torch.nn.Dropout(0.25)

            self.relu = torch.nn.ReLU()

            self.projection_image = Sequential( 
                Linear(768, 1024),
                # self.relu,
                # self.dropout0,
                # Linear(1024, 1024),
                # self.relu
            )
            self.projection_text = Sequential( 
                Linear(768, 1024),
                # self.relu,
                # self.dropout0,
                # Linear(1024, 1024),
                # self.relu
            )
            
            # self.projection_hate1 = Linear(768, 512)
            # self.projection_hate2 = Linear(768, 512)
            self.text_image_net = Sequential(
                self.dropout0,
                Linear(1024, 1024),
                self.relu,
                self.dropout0,
                Linear(1024, 1024),
                self.relu,
                self.dropout0,
                Linear(1024, 1024),
                self.relu,
                self.dropout0,
                Linear(1024, 2),
            )



        def forward(self, text, image):

            text_projection = self.projection_text(text)
            image_projection = self.projection_image(image)

            image_projection = torch.nn.functional.normalize(image_projection, p=2, dim=1)
            text_projection = torch.nn.functional.normalize(text_projection, p=2, dim=1)



            text_image = torch.mul(image_projection, text_projection)

            # hate1 = self.projection_hate1(self.hate_tensor)
            # hate2 = self.projection_hate2(self.hate_tensor)

            # hate_align1 = torch.mul(text_projection, hate1)
            # hate_align2 = torch.mul(image_projection, hate2)

            # ful = torch.cat([text_image, hate_align1, hate_align2], dim=1)

            out = self.text_image_net(text_image)

            return out
