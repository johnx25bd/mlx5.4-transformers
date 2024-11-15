import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_keys=None):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_keys = d_keys if d_keys is not None else d_model
        
        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)  # Query dim stays the same
        self.W_K = nn.Linear(self.d_keys, d_model)  # Project keys to d_model
        self.W_V = nn.Linear(self.d_keys, d_model)  # Project values to d_model
        self.W_O = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        
        # Initialize biases as zeros
        nn.init.constant_(self.W_Q.bias, 0.)
        nn.init.constant_(self.W_K.bias, 0.)
        nn.init.constant_(self.W_V.bias, 0.)
        nn.init.constant_(self.W_O.bias, 0.)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0) if len(Q.size()) > 2 else 1
        seq_len_q = Q.size(-2) if len(Q.size()) > 2 else Q.size(0)
        seq_len_k = K.size(-2) if len(K.size()) > 2 else K.size(0)
        
        # Add batch dimension if not present
        if len(Q.size()) == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
        
        # Linear projections and split into heads
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(context)
        
        # Remove batch dimension if it was added
        if batch_size == 1 and len(Q.size()) == 3:
            output = output.squeeze(0)
            
        return output