import tomli

import random
import pandas as pd

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

import wandb
from tqdm import tqdm

from core import SimpleTransformer
from utils.tokenizer import load_spm_model
from utils.data import load_data

# Set hyperparameters
with open('HYPERPARAMETERS.toml', 'rb') as f:
    hyperparams = tomli.load(f)

# HYPERPARAMETERS
# Sentencepiece
VOCAB_SIZE = hyperparams['sentencepiece']['vocab_size']
EMB_DIM = hyperparams['sentencepiece']['emb_dim']
CHARACTER_COVERAGE = hyperparams['sentencepiece']['character_coverage']
SPM_MODEL_TYPE = hyperparams['sentencepiece']['spm_model_type']

# Model
MAX_SEQ_LEN = hyperparams['model']['max_seq_len']
# Training
LEARNING_RATE = hyperparams['training']['learning_rate']
BATCH_SIZE = hyperparams['training']['batch_size']
# Environment
SEED = hyperparams['environment']['seed']
DEVICE = hyperparams['environment']['device']

torch.manual_seed(SEED)
torch.device(DEVICE)


# # Import tokenizer
spm_processor, model_path = load_spm_model(
                                VOCAB_SIZE, 
                                CHARACTER_COVERAGE, 
                                SPM_MODEL_TYPE)

# # Prepare data
data = load_data("./data/train-sample.txt")
# print(data[0:2])
print(spm_processor.encode(data, out_type=str))

# # instantiate dataset
# # instantiate dataloader with collate
# # confirm data is batched correctly

# # Instantiate model
# Instantiate model
model = SimpleTransformer(VOCAB_SIZE, EMB_DIM)
model.to(DEVICE)


# # Instantiate optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# # Instantiate loss function
# criterion = nn.CrossEntropyLoss()

# # Train model
# wandb.init(project="mlx5.4-transformers", name="simple-transformer-alphabet")
# model.train()
# # Debug prints

# for epoch in tqdm(range(1500)):
#     for i, (actual, ex, mask) in enumerate(dataloader):
#         logits = model(ex, mask) 
#         logits = logits.view(-1, VOCAB_SIZE)
#         actual = actual.view(-1)

#         loss = F.cross_entropy(logits, actual)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         wandb.log({"loss": loss})
#     # Validate every N batches
#     if epoch % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             # Get predictions for first sequence in batch
#             pred = torch.argmax(logits.view(ex.shape[0], -1, VOCAB_SIZE)[0], dim=-1)
#             act = actual.view(ex.shape[0], -1)[0]
            
#             pred_tokens = ''.join([id2token[i.item()] for i in pred])
#             actual_tokens = ''.join([id2token[i.item()] for i in act])
            
#             print(f"\nEpoch {epoch}, Batch {i}")
#             print(f"Predicted: {pred_tokens}")
#             print(f"Actual:    {actual_tokens}")
#         torch.save(model, f"models/model-{epoch}.pt")
#         model.train()
# wandb.finish()