import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_keys=None, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"

        # Define dimensions first
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_keys = d_keys if d_keys is not None else d_model
        
        # Now we can safely register the scaling buffer
        self.register_buffer("scaling", torch.sqrt(torch.FloatTensor([self.d_k])))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(self.d_keys, d_model)
        self.W_V = nn.Linear(self.d_keys, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Dropouts with recommended rates
        self.attention_dropout = nn.Dropout(p=dropout)
        self.output_dropout = nn.Dropout(p=dropout)
        
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
        
        # Add batch dimension if not present
        if len(Q.size()) == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
        
        # Apply layer norm first (pre-norm formulation)
        Q = self.layer_norm(Q)
        
        # Linear projections and split into heads
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling
        
        # Apply mask if provided
        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(context)
        output = self.output_dropout(output)
        
        # Remove batch dimension if it was added
        if batch_size == 1 and len(Q.size()) == 3:
            output = output.squeeze(0)
            
        return output