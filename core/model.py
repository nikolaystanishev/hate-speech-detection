import torch
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
