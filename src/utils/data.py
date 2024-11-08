import re
import random
import pickle
import pandas as pd

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