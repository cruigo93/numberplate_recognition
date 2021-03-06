from torch import nn
import torch.nn.functional as F
import pretrainedmodels

class TextRecogModel(nn.Module):
    def __init__(self, base, num_chars):
        super(TextRecogModel, self).__init__()
        self.base = base
        self.num_chars = num_chars
        
        self.linear = nn.Linear(10240, 64)
        self.dropout = nn.Dropout(0.2)
        
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2)
        self.output = nn.Linear(64, num_chars+1)
        
        
        
    def forward(self, img):
        bs, ch, h, w = img.shape
        features = self.base.extract_features(img)
        # print(features.shape)
        features = features.permute(0, 3, 1, 2)
        # print(features.shape)
        features = features.contiguous().view(bs, features.size(1), -1)
        # print(features.shape)
        x = self.linear(features)
        x = self.dropout(x)
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)
        # print(x.shape)
        return x