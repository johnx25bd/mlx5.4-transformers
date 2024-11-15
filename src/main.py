import wandb
import torch
import subprocess
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

from mnist import Combine, \
    ImageEncoder, \
    LabelEncoder, \
    ImageLabelingModel

torch.manual_seed(42)

SMALLER_PATCH_SIZE_14 = 14
OVERALL_GRID_SIZE_16 = 16

def prep_img(img, 
             smaller_patch_size=SMALLER_PATCH_SIZE_14, 
             overall_grid_size=OVERALL_GRID_SIZE_16):

    img = np.array(img).reshape(overall_grid_size, 
                                smaller_patch_size, 
                                smaller_patch_size)
    flattened_patches = [patch.flatten() for patch in img]
    flattened_patches = np.array(flattened_patches)
    flattened_patches = torch.tensor(flattened_patches, dtype=torch.float32)
    
    return flattened_patches

def test_image_encoder(learning_rate=0.000001, 
                       num_examples=10000):

    ds = Combine()
    img_encoder = ImageEncoder(patch_pixel_num=196, 
                                     img_emb_dim=64)
    
    img_loss_fn = nn.MSELoss()
    img_optim = torch.optim.Adam(img_encoder.parameters(), lr=learning_rate)
    img_actual_zeroes = torch.zeros(size=(16, 64))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project="mlx5.4-transformers", name=f"py-image-encoder-test-7bce240-{timestamp}")

    for i in range(num_examples):

        test_img = ds[i][0]
        test_flattened_patches = prep_img(test_img)

        img_encoding = img_encoder(test_flattened_patches)

        img_loss = img_loss_fn(img_encoding, img_actual_zeroes)
        img_optim.zero_grad()
        img_loss.backward()
        img_optim.step()

        wandb.log({
            "img_loss": img_loss.item()
        })
        if i % 1000 == 0:
            print(f"""
                  Iteration {i} completed
                  img_loss: {img_loss.item()}""")
    # TEST IMAGE ENCODER END!

    wandb.finish()

def test_label_encoder(learning_rate=0.001, 
                       num_examples=1000):
    
    ds = Combine()
    ex1 = [10] + ds[0][1]
    print('ex1.type')
    ex1 = torch.LongTensor(ex1)
    # ex2 = ds[1][0] + [11]
    # ex2 = torch.tensor(ex2, dtype=torch.float32)

    label_encoder = LabelEncoder(label_emb_dim=32, 
                                 vocab_size=12,
                                 num_atn_blocks=8)
    
    label_loss_fn = nn.CrossEntropyLoss()
    label_optim = torch.optim.Adam(label_encoder.parameters(), lr=learning_rate)
    label_actual_ones = torch.ones(size=(5, 32))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project="mlx5.4-transformers", name=f"py-label-encoder-test-7bce240-{timestamp}")

    for i in range(num_examples):
        test_label = ex1.clone()
        # test_label = torch.tensor(test_label, dtype=torch.int64)
        label_encoding = label_encoder(test_label)

        label_loss = label_loss_fn(label_encoding, label_actual_ones.clone())
        label_optim.zero_grad()
        label_loss.backward()
        label_optim.step()

        wandb.log({
            "label_loss": label_loss.item()
        })
        if i % 1000 == 0:
            print(f"""
                  Iteration {i} completed
                  label_loss: {label_loss.item()}""")
    # TEST LABEL ENCODER END!

    wandb.finish()
    label_encoder.eval()
    pred = label_encoder(ex1)
    print('example:', label_actual_ones)
    print('pred:', pred)

def train(num_epochs=10, num_examples=1):
    
    
    """
    [x] Train: 100000 epochs, 1 example, 8 (x)atn blocks, patch_pixel_num=196, img_emb_dim=64, label_emb_dim=32, vocab_size=12
        - Converged after ~350 epochs
        - loss: 1.60944
        - accuracy: 0.2 (random)
    [x] Train: 100000 epochs, 1 example, as above, plus normalization + residual connections
        - Converged a bit more quickly
        - Loss: almost identical
        - Accuracy: 0.2 (random), with a few odd periods of 40% 🤔
        - Adding in step decay to learning rate
    [x] Train: 100000 epochs, 1 example, as above, plus normalization + residual connections, plus step decay
        - Converged after ~250 epochs, then at 15815 dropped off again massively
        - Loss mirrored prior behavior, then dropped off again quickly. Second convergence at 0.666
        - Accuracy rose between epoch 10k and 15k, to 0.6, then at 35727 jumped to 0.8
        - Weird behavior, but it's learning ... ? 🤷‍♂️
    [x] Train: 100000 epochs, 1 example, as above, plus normalization + residual connections, plus step decay, plus linear projection in attention
        - Strange behavior, immediately dropped to loss of near zero, then up to 1.6, then after ~35k examples loss began to slowly drop
    [x] Train: 100000 epochs, 1 example, as above, plus normalization + residual connections, plus step decay, plus linear projection in attention, plus dropout
        - Quickly (3000 examples) converged to a loss of 1.6, then stayed pretty constant
    [x] FIX: Fixed bug in attention block, now converges quickly
    [x] Train: 50 epochs, 2000 examples 😳
        - Loss kep dropping but did not see improvement in predictions, which seems odd? 
    [ ] Train: 50 epochs, 2000 examples, as above, plus normalization + residual connections, plus step decay, plus linear projection in attention, plus dropout, plus positional encoding
    [ ] Train: 50 epochs, 2000 examples, as above, plus normalization + residual connections, plus step decay, plus linear projection in attention, plus dropout, plus positional encoding, plus multi-head attention
    """
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    ds = Combine()
    model = ImageLabelingModel(patch_pixel_num=196, 
                              img_emb_dim=64, 
                              label_emb_dim=32, 
                              vocab_size=12,
                              num_atn_blocks=8)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=0.7, 
                                  patience=10,
                                  threshold=0.01,
                                  cooldown=5,
                                  min_lr=1e-8,
                                  verbose=True)
    
    wandb.init(project="mlx5.4-transformers", 
               name=f"py-image-encoder-{commit_hash}-{timestamp}")

    epoch_loss = 0
    for epoch in range(num_epochs):
        for i in range(num_examples):
            
            orig_img, orig_label = ds[i]  

            actual = torch.LongTensor(orig_label.copy() + [11])
            label = torch.LongTensor([10] + orig_label.copy())
            
            img_flattened = prep_img(orig_img)
            logits = model(img_flattened, label)

            loss = loss_fn(logits, actual)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "loss": loss.item(),
                "accuracy": (logits.argmax(dim=1) == actual).float().mean(),
                "lr": optimizer.param_groups[0]['lr']
            })
            epoch_loss += loss.item()
            if epoch % 20000 == 0:
                print(f'epoch: {epoch}, example: {i}')
                # orig_img.show()
                print('Actual:', label)
                print('Pred:', logits.argmax(dim=1))
                print("loss:", loss.item())
        
        if epoch % 100 == 0:
            avg_epoch_loss = epoch_loss / 100
            scheduler.step(avg_epoch_loss)
            epoch_loss = 0
            if epoch % 10000 == 0:
                print(f'Epoch {epoch} completed, avg_loss: {avg_epoch_loss}')
    wandb.finish()


    

if __name__ == "__main__":
    train(2000, 50)