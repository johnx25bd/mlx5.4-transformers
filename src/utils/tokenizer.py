import os
import tomllib
import pandas as pd
from datasets import load_dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor



def load_hyperparameters(filepath):
    with open(filepath, 'rb') as f:
        hyperparameters = tomllib.load(f)
    vocab_size = hyperparameters['sentencepiece']['vocab_size']
    character_coverage = hyperparameters['sentencepiece']['character_coverage']
    spm_model_type = hyperparameters['sentencepiece']['spm_model_type']

    return (hyperparameters, vocab_size, character_coverage, spm_model_type)

def build_spm_model_prefix(vocab_size, 
                           character_coverage, 
                           spm_model_type):
    return f"spm_vs-{vocab_size}_cc-{character_coverage}_mt-{spm_model_type}"

def load_spm_model(vocab_size, character_coverage, spm_model_type):

    model_prefix = build_spm_model_prefix(
                    vocab_size, 
                    character_coverage, 
                    spm_model_type)
    model_path = f"./tokenizer/models/{model_prefix}.model"

    # Check if model path exists
    if os.path.exists(model_path):
        try:
            spm_model = SentencePieceProcessor(model_file=model_path)
            print(f"Loaded pretrained SentencePiece model from {model_path}")
            return spm_model, model_path
        except Exception as e:
            print(f"Error loading SentencePiece model: {e}")
            return None
    else:
        print(f"""
            No pretrained SentencePiece model found at {model_path}
            Downloading wikitext-2-v1 dataset...

            """)
        try:
            dataset = load_dataset("wikitext", "wikitext-2-v1")
            train_dataset = dataset["train"]

            import tempfile
            # Create temporary file for training
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                temp_file.write('\n'.join(train_dataset['text']))
                temp_file_path = temp_file.name


            SentencePieceTrainer.train(
                input=temp_file_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=character_coverage,
                model_type=spm_model_type
            )

            os.remove(temp_file_path)

            spm_model = SentencePieceProcessor(model_file=model_path)
            return spm_model, model_path
        except Exception as e:
            print(f"Error training SentencePiece model: {e}")
            return None


def main():

    (_, 
    VOCAB_SIZE, 
    CHARACTER_COVERAGE, 
    SPM_MODEL_TYPE) = load_hyperparameters('./HYPERPARAMETERS.toml')

    (spm_processor, 
     model_path) = load_spm_model(
                        VOCAB_SIZE, 
                        CHARACTER_COVERAGE, 
                        SPM_MODEL_TYPE)

    data_str = "This is a test sentence."
    encoded_data = spm_processor.encode(data_str, out_type=str)
    print(encoded_data)

if __name__ == "__main__":
    main()