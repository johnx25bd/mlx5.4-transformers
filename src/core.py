import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        # LINEAR LAYER to use inside attention mechanism!?!?!
        self.out = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, x, padding_mask=None):

        x = self.emb(x)

        # Here we'd do positional encoding :P
        scores = x @ x.transpose(-2, -1)
        # print("\nAttention scores before mask:", 
        #       "\nmin:", scores.min().item(),
        #       "\nmax:", scores.max().item())

        if padding_mask is not None:
            attention_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
            # print("\nAttention mask sample:")
            # print(attention_mask[0, :3, :3])  # Show first 3x3 of first batch
            
            scores = scores.masked_fill(~attention_mask, -1e9)
            scores = scores / math.sqrt(x.size(-1))
            # print("\nAttention scores after mask:", 
            #       "\nmin:", scores.min().item(),
            #       "\nmax:", scores.max().item())
        
        
        atn = F.softmax(scores, -1)
        # print("\nAttention weights after softmax:",
        #       "\nmin:", atn.min().item(),
        #       "\nmax:", atn.max().item())
        
        enc = atn @ x
        logits = self.out(enc)
        # print("\nLogits:",
        #       "\nmin:", logits.min().item(),
        #       "\nmax:", logits.max().item(),
        #       "\nShape:", logits.shape)
        
        # probs = F.softmax(logits, dim=-1)
        return logits

