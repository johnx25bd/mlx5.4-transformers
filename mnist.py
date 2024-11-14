import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, emb_size):
        super(Attention, self).__init__()
        self.emb_size = emb_size
        self.W_Q = nn.Linear(self.emb_size, self.emb_size)
        self.W_K = nn.Linear(self.emb_size, self.emb_size)
        self.W_V = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, encoding):
        Q = self.W_Q(encoding)
        K = self.W_K(encoding)
        V = self.W_V(encoding)

        atn_scores = Q @ K.T
        atn_weights = F.softmax(atn_scores, dim=-1)
        atn_output = atn_weights @ V
        return atn_output

class CrossAttention(nn.Module):
    def __init__(self, img_emb_dim, label_emb_dim, x_emb_dim=56):
        super(CrossAttention, self).__init__()
        self.img_emb_dim = img_emb_dim
        self.label_emb_dim = label_emb_dim
        self.x_emb_dim = x_emb_dim

        self.W_QX = nn.Linear(self.label_emb_dim, self.x_emb_dim)
        self.W_KX = nn.Linear(self.img_emb_dim, self.x_emb_dim)
        self.W_VX = nn.Linear(self.img_emb_dim, self.label_emb_dim)

        self.x_ff = nn.Sequential(
            nn.Linear(self.label_emb_dim, self.label_emb_dim), # project up?
            nn.ReLU(),
            nn.Linear(self.label_emb_dim, self.label_emb_dim)
        )

    def forward(self, label_encoding, img_encoding):
        qx = self.W_QX(label_encoding)
        kx = self.W_KX(img_encoding)
        vx = self.W_VX(img_encoding)

        xatn_scores = qx @ kx.T
        xatn_weights = F.softmax(xatn_scores, dim=-1)
        xatn_output = xatn_weights @ vx

        xatn_output = self.x_ff(xatn_output)

        return xatn_output # image-enriched label encoding

class ImageEncoder(nn.Module):
    def __init__(self, patch_pixel_num=196, img_emb_dim=64):
        super(ImageEncoder, self).__init__()
        self.patch_pixel_num = patch_pixel_num
        self.img_emb_dim = img_emb_dim
        self.linear_layer = nn.Linear(self.patch_pixel_num, 
                                      self.img_emb_dim)


        self.atn_blocks = [Attention(self.img_emb_dim) for _ in range(1)]

        self.img_ff = nn.Sequential(
            nn.Linear(self.img_emb_dim, self.img_emb_dim),
            nn.ReLU(),
            nn.Linear(self.img_emb_dim, self.img_emb_dim)
        )
    
    def forward(self, x):
        img_embedding = self.linear_layer(x)
        for atn_block in self.atn_blocks:
            x = atn_block(img_embedding)
        img_encoding = self.img_ff(img_embedding)
        return img_encoding

class LabelEncoder(nn.Module):
    def __init__(self, label_emb_dim=32, vocab_size=12):
        super(LabelEncoder, self).__init__()
        self.label_emb_dim = label_emb_dim
        self.vocab_size = vocab_size
        self.label_embedding_matrix = nn.Embedding(self.vocab_size, self.label_emb_dim)

        self.atn_blocks = [Attention(self.label_emb_dim) for _ in range(1)]

    def forward(self, labels):
        label_encoding = self.label_embedding_matrix(labels)
        for atn_block in self.atn_blocks:
            label_encoding = atn_block(label_encoding)
        return label_encoding

class ImageCaptioningModel(nn.Module):
    def __init__(self, patch_pixel_num=196, img_emb_dim=64, label_emb_dim=32, vocab_size=12):
        super(ImageCaptioningModel, self).__init__()
        self.ImageEncoder = ImageEncoder(patch_pixel_num, img_emb_dim)
        self.LabelEncoder = LabelEncoder(label_emb_dim, vocab_size)
        self.CrossAttention_blocks = [CrossAttention(self.ImageEncoder.img_emb_dim, 
                                               self.LabelEncoder.label_emb_dim, 
                                               x_emb_dim=56) for _ in range(1)]
        
        self.projection_layer = nn.Linear(self.LabelEncoder.label_emb_dim, self.LabelEncoder.vocab_size)
    
    def forward(self, image, label):
        img_encoding = self.ImageEncoder(image)
        label_encoding = self.LabelEncoder(label)
        for cross_atn in self.CrossAttention_blocks:
            label_encoding = cross_atn(label_encoding, img_encoding)
        logits = self.projection_layer(label_encoding)
        return logits

if __name__ == "__main__":
    model = ImageCaptioningModel(patch_pixel_num=196, img_emb_dim=64, label_emb_dim=32, vocab_size=12)
    print(model)