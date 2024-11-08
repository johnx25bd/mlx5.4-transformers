import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, emb_dim):
        super(Attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.scale = math.sqrt(emb_dim)

    def forward(self, x, padding_mask=None): # How to handle padding mask?
        proj = self.linear(x) # (batch_size, seq_len, emb_dim)
        atn = (proj @ proj.transpose(-2, -1)) / self.scale # (batch_size, seq_len, seq_len)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
            atn = atn.masked_fill(~mask, -1e9)
        atn = F.softmax(atn, -1)
        out = atn @ proj # (batch_size, seq_len, emb_dim)
        return out




class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_blocks=2):
        super(SimpleTransformer, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.magics = nn.ModuleList([Attention(emb_dim) for _ in range(num_blocks)])
        self.vocab = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, x, padding_mask=None):

        # Here we'd do positional encoding? Or prior to input? :P
        embs = self.emb(x)
        for magic in self.magics:
            embs = magic(embs, padding_mask)
        logits = self.vocab(embs)
        # Softmax? crossentropy loss expects logits ...
        return logits
   

        # if padding_mask is not None:
        #     attention_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)

        #     scores = scores.masked_fill(~attention_mask, -1e9)
        #     scores = scores / math.sqrt(x.size(-1))
        
        # atn = F.softmax(scores, -1)
        # enc = atn @ x
        # logits = self.out(enc)
        
        # return logits

