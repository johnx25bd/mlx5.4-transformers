import tomli

def load_hyperparameters(filepath):
# Set hyperparameters
    with open(filepath, 'rb') as f:
        hyperparams = tomli.load(f)

    # HYPERPARAMETERS
    PROJECT_NAME = hyperparams['project']['name']
    PROJECT_VERSION = hyperparams['project']['version']
    # Sentencepiece
    VOCAB_SIZE = hyperparams['sentencepiece']['vocab_size']
    EMB_DIM = hyperparams['sentencepiece']['emb_dim']
    CHARACTER_COVERAGE = hyperparams['sentencepiece']['character_coverage']
    SPM_MODEL_TYPE = hyperparams['sentencepiece']['spm_model_type']

    # Model
    MAX_SEQ_LEN = hyperparams['model']['max_seq_len']
    MODEL_NAME = hyperparams['model']['model_name']
    # Training
    LEARNING_RATE = hyperparams['training']['learning_rate']
    BATCH_SIZE = hyperparams['training']['batch_size']
    # Environment
    SEED = hyperparams['environment']['seed']
    DEVICE = hyperparams['environment']['device']

    return {
        'PROJECT_NAME': PROJECT_NAME,
        'PROJECT_VERSION': PROJECT_VERSION,
        'VOCAB_SIZE': VOCAB_SIZE,
        'EMB_DIM': EMB_DIM,
        'CHARACTER_COVERAGE': CHARACTER_COVERAGE,
        'SPM_MODEL_TYPE': SPM_MODEL_TYPE,
        'MAX_SEQ_LEN': MAX_SEQ_LEN,
        'MODEL_NAME': MODEL_NAME,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'SEED': SEED,
        'DEVICE': DEVICE
    }