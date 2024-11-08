import re
import random
import pickle
import pandas as pd
import torch


# Read in text data from path
def load_data(path):
    with open(path, 'r') as file:
        text = file.read()
    
    return text


def preprocess_text(text):
    # Split into articles, delimited by  = <Title> =  (single = characters)
    pattern = r'^\s*=\s[^=].*[^=]\s=$'
    
    # Split the text on matching lines
    sections = re.split(pattern, text, flags=re.MULTILINE)
    
    return sections

def letter_sequences(num_examples, alphabet, max_seq_len, min_seq_len=3, seed=42):

    examples = []
    for i in range(num_examples):
        # Set seed for reproducibility
        random.seed(seed * i)
        length = random.randint(min_seq_len, max_seq_len)
        start = random.randint(0, len(alphabet) - length)
        example = alphabet[start:start+length]
        examples.append(example)
    return pd.Series(examples)

def generate_alphabet_sequences(num_examples=10000, 
                            alphabet="abcdefghijklmnopqrstuvwxyz", 
                            min_seq_len=3, 
                            max_seq_len=10):
    data = pd.DataFrame()
    data["sequence"] = letter_sequences(num_examples, alphabet, max_seq_len, min_seq_len)
    return data

def apply_mask(row):
    # mask is a list of 0s and 1s
    return [a * m for a, m in zip(row['actual'], row['mask'])]
def tokenize_alphabet_sequences(dataframe, vocab2id):
    dataframe["actual"] = dataframe["sequence"].apply(lambda x: [vocab2id[token] for token in x])
    return dataframe

def create_examples(dataframe, vocab2id):
    dataframe["mask"] = dataframe["actual"].apply(lambda x: [1 if random.random() > 0.15 else 0 for _ in x])
    dataframe["example"] = dataframe.apply(apply_mask, axis=1)
    return dataframe



"""Accepts a string or list of tokens.
    Returns id2token and token2id dictionaries.
"""
def vocab_dicts(vocab):
    vocab = sorted(set(vocab))
    id2token = ["<UNK>", *vocab]
    token2id = {token: i for i, token in enumerate(id2token)}
    return id2token, token2id



##### TORCH CLASSES #####

class AlphabetDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab_size, mask_prob=0.15):
        self.data = data
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token = 0
        self.pad_token_id = -100
        self.mask_token_id = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sequence = torch.tensor(self.data.iloc[idx]['actual'])
        # Create masked input (x) and target (y)
        x = sequence.clone()
        
        # Randomly select positions to mask
        mask_positions = torch.rand(len(sequence)) < self.mask_prob
        
        # Apply masking strategy
        for pos in range(len(sequence)):
            if mask_positions[pos]:
                rand = random.random()
                if rand < 0.8:  # 80% mask token
                    x[pos] = self.mask_token_id
                elif rand < 0.9:  # 10% random token
                    x[pos] = random.randint(1, self.vocab_size-1)
                # else: 10% keep unchanged
        
        return sequence, x, mask_positions

def collate(batch, pad_token_id=-100):

    sequences, inputs, _ = zip(*batch)

    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                       batch_first=True, 
                                                       padding_value=pad_token_id)
    padded_inputs =  torch.nn.utils.rnn.pad_sequence(inputs, 
                                                     batch_first=True, 
                                                     padding_value=pad_token_id)

    attention_mask = padded_sequences != pad_token_id

    return padded_sequences, padded_inputs, attention_mask
    




if __name__ == "__main__":
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    data = generate_alphabet_sequences()
    id2token, token2id = vocab_dicts(alphabet)
    data = tokenize_alphabet_sequences(data, token2id)
    data = create_examples(data, token2id)
    data.to_parquet("../data/tokenized_alphabet_sequences.parquet")
    # Save id2token and token2id to pickle
    with open("../data/alphabet_id2token.pkl", "wb") as f:
        pickle.dump(id2token, f)
    with open("../data/alphabet_token2id.pkl", "wb") as f:
        pickle.dump(token2id, f)