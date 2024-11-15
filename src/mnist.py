import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image

random.seed(42)

### DATA ###
class Combine(torch.utils.data.Dataset):
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(42)
        self.tf = transforms.ToTensor()
        self.ds = datasets.MNIST(root='./data', train=True, transform=self.tf, download=True)

    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, idx):
        idx = random.sample(range(len(self)), 4)
        store = []
        label = []

        for i in idx:
            x, y = self.ds[i]
            x = transforms.ToPILImage()(x).convert('L')  # Ensure mode 'L'

            store.append(x)
            label.append(y)

        img = Image.new('L', (56, 56))

        img.paste(store[0], (0, 0, 28, 28))      # top-left
        img.paste(store[1], (28, 0, 56, 28))     # top-right
        img.paste(store[2], (0, 28, 28, 56))     # bottom-left
        img.paste(store[3], (28, 28, 56, 56))    # bottom-right
        
        return img, label

### POSITIONAL ENCODING ###
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=16):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(1000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


### ATTENTION ###
class Attention(nn.Module):
    def __init__(self, emb_size):
        super(Attention, self).__init__()
        self.emb_size = emb_size
        self.W_Q = nn.Linear(self.emb_size, self.emb_size)
        self.W_K = nn.Linear(self.emb_size, self.emb_size)
        self.W_V = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.emb_size)
        self.project = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, encoding):
        Q = self.W_Q(encoding)
        K = self.W_K(encoding)
        V = self.W_V(encoding)

        atn_scores = Q @ K.T
        atn_weights = F.softmax(atn_scores, dim=-1)
        atn_output = atn_weights @ V
        atn_output = self.dropout(atn_output)
        atn_output = self.norm(atn_output + encoding) # Added residual connection + normalization!
        # Bes projects here!
        out = self.project(atn_output)
        return out # return out

class MaskedAttention(Attention):
    def __init__(self, emb_size):
        super(Attention, self).__init__()
        self.emb_size = emb_size
        self.W_Q = nn.Linear(self.emb_size, self.emb_size)
        self.W_K = nn.Linear(self.emb_size, self.emb_size)
        self.W_V = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.emb_size)
        self.project = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, encoding):
        Q = self.W_Q(encoding)
        K = self.W_K(encoding)
        V = self.W_V(encoding)

        atn_scores = Q @ K.T

        negative_inf = torch.full_like(atn_scores, float('-inf'))
        mask = torch.triu(negative_inf, diagonal=1)

        masked_atn_scores = atn_scores + mask

        atn_weights = F.softmax(masked_atn_scores, dim=-1)
        atn_output = atn_weights @ V
        atn_output = self.dropout(atn_output)
        atn_output = self.norm(atn_output + encoding) # Added residual connection + normalization!
        out = self.project(atn_output)
        return out

class CrossAttention(nn.Module):
    def __init__(self, img_emb_dim, label_emb_dim, x_emb_dim=56):
        super(CrossAttention, self).__init__()
        self.img_emb_dim = img_emb_dim
        self.label_emb_dim = label_emb_dim
        self.x_emb_dim = x_emb_dim

        self.W_QX = nn.Linear(self.label_emb_dim, self.x_emb_dim)
        self.W_KX = nn.Linear(self.img_emb_dim, self.x_emb_dim)
        self.W_VX = nn.Linear(self.img_emb_dim, self.label_emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.x_ff = FeedForward(self.label_emb_dim, self.label_emb_dim)
        self.norm = nn.LayerNorm(self.label_emb_dim)
        self.project = nn.Linear(self.label_emb_dim, self.label_emb_dim)
    def forward(self, label_encoding, img_encoding):
        qx = self.W_QX(label_encoding)
        kx = self.W_KX(img_encoding)
        vx = self.W_VX(img_encoding)

        xatn_scores = qx @ kx.T
        xatn_weights = F.softmax(xatn_scores, dim=-1)
        xatn_output = xatn_weights @ vx
        xatn_output = self.dropout(xatn_output)
        xatn_output = self.x_ff(xatn_output)
        xatn_output = self.norm(xatn_output + label_encoding) # Added residual connection + normalization!
        x_out = self.project(xatn_output)
        return x_out # image-enriched label encoding

### FEEDFORWARD ###
class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.l1 = nn.Linear(self.emb_dim, self.ff_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.ff_dim, self.emb_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

### ENCODERS ###
class ImageEncoder(nn.Module):
    def __init__(self, patch_num=16, patch_pixel_num=196, img_emb_dim=64):
        super(ImageEncoder, self).__init__()
        self.patch_num = patch_num
        self.patch_pixel_num = patch_pixel_num

        self.positional_encoding = PositionalEncoding(self.patch_num)

        self.img_emb_dim = img_emb_dim
        self.linear_layer = nn.Linear(self.patch_pixel_num, 
                                      self.img_emb_dim)


        self.atn_blocks = nn.ModuleList([Attention(self.img_emb_dim) for _ in range(1)])
        self.project = nn.Linear(self.img_emb_dim, self.img_emb_dim)
        self.img_ff = FeedForward(self.img_emb_dim, 
                                  self.img_emb_dim * 4)
    
    def forward(self, x):
        x = self.positional_encoding(x)
        img_embedding = self.linear_layer(x)
        for atn_block in self.atn_blocks:
            img_embedding = atn_block(img_embedding)
        img_encoding = self.project(img_embedding)
        img_encoding = self.img_ff(img_embedding)

        return img_encoding
    
class LabelEncoder(nn.Module):
    def __init__(self, label_len=5, label_emb_dim=32, vocab_size=12, num_atn_blocks=5):
        super(LabelEncoder, self).__init__()
        self.label_len = label_len
        self.positional_encoding = PositionalEncoding(self.label_len)
        self.label_emb_dim = label_emb_dim
        self.vocab_size = vocab_size
        self.label_embedding_matrix = nn.Embedding(self.vocab_size, self.label_emb_dim)

        self.atn_blocks = nn.ModuleList([Attention(self.label_emb_dim) for _ in range(num_atn_blocks)])

    def forward(self, labels):
        labels = self.positional_encoding(labels)
        label_encoding = self.label_embedding_matrix(labels)
        for atn_block in self.atn_blocks:
            label_encoding = atn_block(label_encoding)
        
        return label_encoding


### MODEL ###

class ImageLabelingModel(nn.Module):
    def __init__(self, patch_pixel_num=196, 
                 img_emb_dim=64, 
                 label_emb_dim=32, 
                 vocab_size=12, 
                 num_atn_blocks=5,
                 num_xatn_blocks=5):
        super(ImageLabelingModel, self).__init__()
        self.ImageEncoder = ImageEncoder(patch_pixel_num, img_emb_dim)
        self.LabelEncoder = LabelEncoder(label_emb_dim, vocab_size, num_atn_blocks)
        self.CrossAttention_blocks = [CrossAttention(self.ImageEncoder.img_emb_dim, 
                                               self.LabelEncoder.label_emb_dim, 
                                               x_emb_dim=56) 
                                               for _ in range(num_xatn_blocks)]
        
        self.project = nn.Linear(self.LabelEncoder.label_emb_dim, 
                                          self.LabelEncoder.vocab_size)
    
    def forward(self, image, label):
        img_encoding = self.ImageEncoder(image)
        label_encoding = self.LabelEncoder(label)
        for cross_atn in self.CrossAttention_blocks:
            label_encoding = cross_atn(label_encoding, img_encoding)
        logits = self.project(label_encoding)
        return logits

if __name__ == "__main__":
    model = ImageLabelingModel(patch_pixel_num=196, img_emb_dim=64, label_emb_dim=32, vocab_size=12)
    print(model)