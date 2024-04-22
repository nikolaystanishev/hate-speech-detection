import torch
from torch.nn import Module, Sequential, Linear
from transformers import BertModel, BertTokenizer, AutoModel
import torchvision.models as models
import os
import sys
import json
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


# class ClipHateMemeModelFreeze(Module):
#         def __init__(self):
#             super(ClipHateMemeModelFreeze, self).__init__()
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             self.class_filename = os.path.join(os.path.dirname(__file__), '../data/classes_clip_s.jsonl')
#             self.classes = torch.tensor([json.loads(jline)['embedding'] for jline in open(self.class_filename, 'r').readlines()]).repeat(1,64,1).to(device=self.device)
#             self.similarity = torch.nn.CosineSimilarity(dim=2)
#             self.projection = Linear(768, 768)
#             self.projection2 = Linear(768, 768)
#             self.c_n = self.classes.shape[0]
#             self.fc1 = Linear(3*self.c_n, 128)
#             self.fc2 = Linear(128, 32)
#             # self.fc3 = Linear(64, 32)
#             # self.fc4 = Linear(32, 16)
#             self.fc12 = Linear(768, 768)
#             self.fc22 = Linear(768, 128)
#             self.fc32 = Linear(128, 128)
#             self.fc5 = Linear(128+32, 32)
#             self.fc6 = Linear(32, 2)
#             self.dropout = torch.nn.Dropout(0.15)
#             self.relu = torch.nn.ReLU()
    
#         def forward(self, text, image):
#             text_image = torch.mul(text, image)

#             out2 = self.relu(self.fc12(text_image))
#             out2 = self.dropout(out2)
#             out2 = self.relu(self.fc22(out2))
#             out2 = self.dropout(out2)
#             out2 = self.relu(self.fc32(out2))

#             text_image = self.projection2(text_image)
#             text_image = self.projection(text_image)
#             classes = self.relu(self.projection(self.classes))
#             text = self.relu(self.projection(text))
#             image = self.relu(self.projection(image))

#             batch_size = text.size(0)
#             similarity_text = self.similarity(classes[:,:batch_size,:], text.repeat(self.c_n,1,1)).permute(1,0)
#             similarity_image = self.similarity(classes[:,:batch_size,:], image.repeat(self.c_n,1,1)).permute(1,0)
#             similarity_combined = self.similarity(classes[:,:batch_size,:], text_image.repeat(self.c_n,1,1)).permute(1,0)
#             combined = torch.cat([similarity_text, similarity_image, similarity_combined], dim=1)

#             out = self.relu(self.fc1(combined))
#             out = self.dropout(out)
#             out = self.relu(self.fc2(out))
#             out = self.dropout(out)
#             # out = self.relu(self.fc3(out))
#             # out = self.dropout(out)
#             # out = self.relu(self.fc4(out))
#             # out = self.dropout(out)
#             out = self.fc5(torch.cat([out, out2], dim=1))
#             out = self.dropout(out)
#             out = self.fc6(out)
#             return out


class ClipHateMemeModelFreeze(Module):
        def __init__(self):
            super(ClipHateMemeModelFreeze, self).__init__()

            self.projection_image = Linear(768, 1028)
            self.projection_text = Linear(768, 1028)

            self.dropout0 = torch.nn.Dropout(0.2)
            self.dropout1 = torch.nn.Dropout(0.4)
            self.dropout2 = torch.nn.Dropout(0.2)

            self.relu = torch.nn.ReLU()

            self.text_image_net = Sequential(
                self.dropout0,
                Linear(1028, 1028),
                self.relu,
                self.dropout2,
                Linear(1028, 2),
            )
            self.image_net = Sequential(
                self.dropout1,
                Linear(1028, 1028),
                self.relu,
                self.dropout2,
                Linear(1028, 2),
            )
            self.text_net = Sequential(
                self.dropout1,
                Linear(1028, 1028),
                self.relu,
                self.dropout2,
                Linear(1028, 2),
            )

        def forward(self, text, image):
            text_projection = self.projection_text(text)
            image_projection = self.projection_image(image)
            text_projection = torch.nn.functional.normalize(text_projection, p=2, dim=1)
            image_projection = torch.nn.functional.normalize(image_projection, p=2, dim=1)

            text_image = torch.mul(image_projection, text_projection)

            out = self.text_image_net(text_image)
            out_image = self.image_net(image_projection)
            out_text = self.text_net(text_projection)

            out = {'text_image': out, 'image': out_image, 'text': out_text}
            return out
