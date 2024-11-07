import os
import torch
import tomli_w
import subprocess
from datetime import datetime

def create_run_id(model_name, hyperparams, timestamp, commit):
    """Create a descriptive run ID including key hyperparameters"""
    # Select key hyperparams to include in the ID
    # You might want to customize these based on your specific use case
        # TODO: Dynamically select key params and prefixes
    key_params = [
        f"lr-{hyperparams.get('learning_rate', '')}",
        f"bs-{hyperparams.get('batch_size', '')}",
        f"hs-{hyperparams.get('hidden_size', '')}" # This won't always be relevant
    ]
    param_str = "_".join(key_params)
    
    # Combine everything into a readable ID
    return f"{model_name}_{param_str}_{timestamp}_{commit}"


def save_checkpoint(model, epoch, model_name, hyperparams, repo, metrics=None):
    """Save a checkpoint of the model and associated metadata
    Args:
        model: The model to save
        epoch: Current epoch number
        model_name: Name of the model
        hyperparams: Dict of hyperparameters
        repo: Git repository name with user/org prefix (e.g. "johnx25bd/mlx")
        metrics: Dict of current metrics (optional)
    """
    # Create unique run ID
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    run_id = create_run_id(model_name, hyperparams, timestamp, commit)
    
    # Create run directory
    run_dir = os.path.join("./checkpoints", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save model checkpoint
    model_path = os.path.join(run_dir, f"epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_class_name': model.__class__.__name__,
        'hyperparams': hyperparams,
        'metrics': metrics or {},
    }, model_path)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'repo': repo,
        'git_commit': commit,
        'epoch': epoch,
        'hyperparams': hyperparams,
        'metrics': metrics or {},
    }
    
    metadata_path = os.path.join(run_dir, 'metadata.toml')
    with open(metadata_path, 'wb') as f:
        tomli_w.dump(metadata, f)

    # Append to global runs log (keeping this as TOML too)
    runs_path = './checkpoints/runs.toml'
    
    # Read existing runs if file exists
    if os.path.exists(runs_path):
        with open(runs_path, 'rb') as f:
            import tomli
            existing_runs = tomli.load(f)
    else:
        existing_runs = {'runs': []}
    
    # Append new run
    existing_runs['runs'].append({
        'run_id': run_id,
        'metadata': metadata
    })
    
    # Write updated runs
    with open(runs_path, 'wb') as f:
        tomli_w.dump(existing_runs, f)


# TODO: Set this up to work with our models
# def load_checkpoint(checkpoint_path, model_class):
#     """Load a checkpoint and return initialized model
#     Args:
#         checkpoint_path: Path to checkpoint file
#         model_class: The model class to instantiate
#     Returns:
#         tuple: (model, checkpoint_data)
#     """
#     checkpoint = torch.load(checkpoint_path)
    
#     # Initialize model with saved hyperparams
#     model = model_class(**checkpoint['hyperparams']) # This isn't gonna work ...
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()  # Set to evaluation mode
    
#     return model, checkpoint


if __name__ == "__main__":
    # Example usage
    class DummyModel(torch.nn.Module):
        def __init__(self, hidden_size=64):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_size, 10)
        
        def forward(self, x):
            return self.linear(x)

    # Example usage
    model = DummyModel(hidden_size=128)
    hyperparams = {
        'hidden_size': 128,
        'learning_rate': 0.001,
        'batch_size': 32,
    }
    metrics = {
        'train_loss': 0.234,
        'val_accuracy': 0.945,
    }

    save_checkpoint(
        model=model,
        epoch=0,
        model_name='dummy_model',
        hyperparams=hyperparams,
        repo='johnx25bd/mlx5.4-transformers',
        metrics=metrics
    )