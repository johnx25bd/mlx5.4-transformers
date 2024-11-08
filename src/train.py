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

from utils.config import load_hyperparameters
from utils.tokenizer import load_spm_model
from utils.data import load_data

# Set hyperparameters
# Import HYPERPARAMETERS
hyperparams = load_hyperparameters("./HYPERPARAMETERS.toml")
VOCAB_SIZE = hyperparams['VOCAB_SIZE']
EMB_DIM = hyperparams['EMB_DIM']
MAX_SEQ_LEN = hyperparams['MAX_SEQ_LEN']
LEARNING_RATE = hyperparams['LEARNING_RATE']
BATCH_SIZE = hyperparams['BATCH_SIZE']
SEED = hyperparams['SEED']
DEVICE = hyperparams['DEVICE']

torch.manual_seed(SEED)
torch.device(DEVICE)
print(hyperparams)

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