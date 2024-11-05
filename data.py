# %%

import tomllib
import pandas as pd
from datasets import load_dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

# %%

with open('HYPERPARAMETERS.toml', 'rb') as f:
    hyperparameters = tomllib.load(f)

VOCAB_SIZE = hyperparameters['sentencepiece']['vocab_size']
MODEL_PREFIX = hyperparameters['sentencepiece']['model_prefix']
MODEL_TYPE = hyperparameters['sentencepiece']['model_type']
CHARACTER_COVERAGE = hyperparameters['sentencepiece']['character_coverage']

# %% 

print(f"""Training SentencePiece model with:
      VOCAB_SIZE: {VOCAB_SIZE}
      MODEL_TYPE: {MODEL_TYPE}
      CHARACTER_COVERAGE: {CHARACTER_COVERAGE}
""")


# %%
# Load wikitext-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")

# %%
train_dataset = dataset["train"]
test_dataset = dataset["test"]
validation_dataset = dataset["validation"]

# %%
sample_data = ['',' aaa ', ' bbb ', ' ccc ']

# %% 
s_data_str = ' '.join(sample_data)
# %%

with open('data/s_train.txt', 'w') as f:
    f.write(s_data_str)

# %%
with open('data/train.txt', 'w') as f:
    f.write('\n'.join(train_dataset['text']))    
# %%
spm_model = SentencePieceTrainer.train(
    input='data/s_train.txt',
    model_prefix="data/s_model",
    vocab_size=10,
    character_coverage=CHARACTER_COVERAGE,
    model_type=MODEL_TYPE
)
# %%
spm = SentencePieceProcessor(model_file='data/s_model.model')
# %%
spm.encode(s_data_str, out_type=str)
# %%
spm_train = SentencePieceTrainer.train(
    input='data/train.txt',
    model_prefix="data/train_model",
    vocab_size=VOCAB_SIZE,
    character_coverage=CHARACTER_COVERAGE,
    model_type=MODEL_TYPE
)
# %%
