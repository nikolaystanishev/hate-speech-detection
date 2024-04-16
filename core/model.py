import torch
import clip
from torch.nn import Module, Sequential, Linear
from transformers import BertModel, BertTokenizer
import torchvision.models as models

class HateMemeModel(Module):

    def __init__(self, text_model, image_model):
        super(HateMemeModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        self.fc1 = Linear(768 + 2048, 1024)
        self.fc2 = Linear(1024, 256)
        self.fc3 = Linear(256, 2)

        self.relu = torch.nn.ReLU()

    def forward(self, text, image):
        text_features = self.relu(self.text_model(**text).pooler_output)
        image_features = self.relu(self.image_model(image))
        image_features = image_features.view(image_features.size(0), -1)
        
        combined = torch.cat([text_features, image_features], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class PretrainedModel:

    @staticmethod
    def load_bert_text_model():
        return BertModel.from_pretrained('bert-base-uncased')
    
    @staticmethod
    def load_bert_tokenizer():
        return BertTokenizer.from_pretrained('bert-base-uncased')
    
    @staticmethod
    def load_resnet_image_model():
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(model.children())[:-1]
        return Sequential(*modules)

    @staticmethod
    def load_clip_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return clip.load("ViT-B/16", device=device)

    @staticmethod
    def load_clip_tokenizer():
        return clip.tokenize

class ClipHateMemeModel(Module):

    def __init__(self, model, tokenizer):
        super(ClipHateMemeModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.fc1 = Linear(512, 32)
        self.fc2 = Linear(64, 16)
        self.fc3 = Linear(16, 2)
        self.dropout = torch.nn.Dropout(0.15)
        self.relu = torch.nn.ReLU()

    def forward(self, text, image):
        text_features = self.model.encode_text(text['input_ids'])
        image_features = self.model.encode_image(image)
        text_features = self.dropout(text_features)
        image_features = self.dropout(image_features)
        text_features = self.relu(self.fc1(text_features.to(dtype=torch.float32)))
        image_features = self.relu(self.fc1(image_features.to(dtype=torch.float32)))
        combined = torch.cat([text_features, image_features], dim=1)
        combined = self.dropout(combined)
        out = self.relu(self.fc2(combined))
        out = self.fc3(out)
        
        return out

