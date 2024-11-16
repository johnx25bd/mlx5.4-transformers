
import torch
from torchvision import datasets, transforms
import random
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from attention import MultiHeadAttention
from loss import LabelSmoothingLoss

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ## Define the transformations to the MINST data

class Combine(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.tf = transform = transforms.Compose([transforms.ToTensor()])
        self.ds = datasets.MNIST(root='./data', train=True, transform=self.tf, download=True)

    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, idx):
        idx = random.sample(range(len(self)), 4)
        store = []
        label = []

        for i in idx:
            x, y = self.ds[i]
            x = transforms.ToPILImage()(x).convert('L')  # Ensure mode 'L'

            store.append(x)
            label.append(y)

        img = Image.new('L', (56, 56))

        img.paste(store[0], (0, 0, 28, 28))      # top-left
        img.paste(store[1], (28, 0, 56, 28))     # top-right
        img.paste(store[2], (0, 28, 28, 56))     # bottom-left
        img.paste(store[3], (28, 28, 56, 56))    # bottom-right
        
        return img, label


minst = Combine()

patch_pixel_num = 196 # should be 196
assert patch_pixel_num == 196
img_emb_dim = 256 # increased from 64
linear_layer = nn.Linear(patch_pixel_num, img_emb_dim)
# W_QI = nn.Linear(img_emb_dim, img_emb_dim) 
# W_KI = nn.Linear(img_emb_dim, img_emb_dim)
# W_VI = nn.Linear(img_emb_dim, img_emb_dim)
img_attention = MultiHeadAttention(d_model=img_emb_dim, num_heads=8, dropout=0.1)



img_ff = nn.Sequential(
    nn.Linear(img_emb_dim, img_emb_dim * 4),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(img_emb_dim * 4, img_emb_dim)
)


id2label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '<s>', 11: '<e>'}
label2id = {v: k for k, v in id2label.items()}


label_emb_size = 128 # increased from 32
vocab_size = len(id2label)
label_embedding_matrix = nn.Embedding(vocab_size, label_emb_size)

# W_QL = nn.Linear(label_emb_size, label_emb_size)
# W_KL = nn.Linear(label_emb_size, label_emb_size)
# W_VL = nn.Linear(label_emb_size, label_emb_size)
label_attention = MultiHeadAttention(d_model=label_emb_size, num_heads=8, dropout=0.1)



x_emb_dim = 192 # increased from 56
# W_QX = nn.Linear(label_emb_size, x_emb_dim)
# W_KX = nn.Linear(img_emb_dim, x_emb_dim)
# W_VX = nn.Linear(img_emb_dim, label_emb_size)
# Initialize with different key dimension
cross_attention = MultiHeadAttention(
    d_model=label_emb_size,  # output dimension (128)
    num_heads=8,
    d_keys=img_emb_dim,      # key/value dimension (256)
    dropout=0.1
)


x_ff = nn.Sequential(
    nn.Linear(label_emb_size, label_emb_size * 4),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(label_emb_size * 4, label_emb_size)
)

# Add layer norms
img_layer_norm = nn.LayerNorm(img_emb_dim)
label_layer_norm = nn.LayerNorm(label_emb_size)
x_layer_norm = nn.LayerNorm(label_emb_size)  # For final encoding


project_layer = nn.Linear(label_emb_size, vocab_size)

# loss_fn = nn.CrossEntropyLoss()
# loss_fn = LabelSmoothingLoss(smoothing=0.1, vocab_size=len(id2label))

# Initialize with more conservative smoothing
loss_fn = LabelSmoothingLoss(
    smoothing=0.05,  # Reduced from 0.1
    vocab_size=len(id2label),
    ignore_index=-100,  
    reduction='mean'
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

# Create fixed positional encodings
img_pos_emb = get_sinusoid_encoding_table(16, img_emb_dim)    # [16, 256]
seq_pos_emb = get_sinusoid_encoding_table(5, label_emb_size)  # [5, 128]


optim = torch.optim.Adam(
    list(label_embedding_matrix.parameters()) +
    list(linear_layer.parameters()) +
    # list(W_QI.parameters()) +
    # list(W_KI.parameters()) +
    # list(W_VI.parameters()) +
    # list(W_QL.parameters()) +
    # list(W_KL.parameters()) +
    # list(W_VL.parameters()) +
    # list(W_QX.parameters()) +
    # list(W_KX.parameters()) +
    # list(W_VX.parameters()) +
    list(img_attention.parameters()) +
    list(label_attention.parameters()) +
    list(cross_attention.parameters()) +
    list(img_ff.parameters()) +
    list(x_ff.parameters()) +
    list(project_layer.parameters()) +
    list(img_layer_norm.parameters()) +  # Add new layer norms
    list(label_layer_norm.parameters()) +
    list(x_layer_norm.parameters()),
    lr=1e-4
)

scheduler = ReduceLROnPlateau(
    optimizer=optim,
    mode='min',
    factor=0.7,
    patience=3,
    verbose=True,
    min_lr=1e-6
)

num_of_epochs = 100
num_examples = 2000

wandb.init(project="mm_transformers_v1", name=f"conservative_label_smoothing_loss_{timestamp}")


for epoch in range(num_of_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0

    for i in range(num_examples):  # how many images to train on?
        
        img, label_store = minst[i]

        # Prepare IMAGE
        grid_size = 14
        overall_grid_num = 16

        img = np.array(img).reshape(overall_grid_num, grid_size, grid_size)
        flattened_patches = torch.tensor(np.array([patch.flatten() for patch in img]), dtype=torch.float32)

        # Prepare LABEL
        label = [10] + label_store
        label = torch.tensor(label)

        actual = label_store + [11]
        actual = torch.tensor(actual)
        # Data PREPARED!

        ### IMAGE ENCODER ###
        img_embeddings = linear_layer(flattened_patches)
        img_embeddings = img_embeddings + img_pos_emb
        norm_img = img_layer_norm(img_embeddings)
        
        # qi = W_QI(norm_img)
        # ki = W_KI(norm_img)
        # vi = W_VI(norm_img)

        # QKI = qi @ ki.T
        # QKI = QKI / torch.sqrt(torch.tensor(img_emb_dim, dtype=torch.float32))
        # softmax_QKI = F.softmax(QKI, dim=-1)

        img_encoding = img_attention(norm_img, norm_img, norm_img) + img_embeddings  # Add residual
        norm_img = img_layer_norm(img_encoding)  # Normalize attention output
        ff_output = img_ff(norm_img)  
        img_encoding = ff_output + img_encoding  # Add residual        

        ### IMAGE ENCODER END! ### 

        ### DECODER ###
        # LABEL ENCODING
        label_embedding = label_embedding_matrix(label)
        assert label_embedding.shape == seq_pos_emb.shape, \
            f"Shape mismatch in decoder: {label_embedding.shape} vs {seq_pos_emb.shape}"
        label_embedding = label_embedding + seq_pos_emb
        norm_label = label_layer_norm(label_embedding)

        # ql = W_QL(norm_label)
        # kl = W_KL(norm_label)
        # vl = W_VL(norm_label)

        # QKL = ql @ kl.T

        # negative_inf = torch.full_like(QKL, float('-inf'))
        # masked_QKL = torch.triu(negative_inf, diagonal=1)

        # Create causal mask for decoder self-attention
        seq_len = norm_label.size(0) # Get current sequence length
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 5, 5]

        # QKL = QKL + masked_QKL

        # QKL = QKL / torch.sqrt(torch.tensor(label_emb_size, dtype=torch.float32))        
        # softmax_QKL = F.softmax(QKL, dim=-1)

        # label_encoding = softmax_QKL @ vl + label_embedding  # Add residual
        # Replace manual label self-attention with multi-head attention
        label_encoding = label_attention(norm_label, norm_label, norm_label, mask=mask) + label_embedding
        norm_label = label_layer_norm(label_encoding)
        # LABEL ENCODING END!

        # CROSS ATTENTION
        norm_label = label_layer_norm(label_encoding)  
        norm_img = img_layer_norm(img_encoding)   
        # qx = W_QX(norm_label)
        # kx = W_KX(norm_img)
        # vx = W_VX(norm_img)

        # QKX = qx @ kx.T # Cross Attention Matrix
        # QKX = QKX / np.sqrt(x_emb_dim) # Scaling
        # softmax_QKX = F.softmax(QKX, dim=-1) # Softmax

        # CROSS ATTENTION
        # x_encoding = softmax_QKX @ vx + label_encoding  # Add residual # Cross Attention Encoding, "image-enriched label encoding"

        # Replace manual cross attention with multi-head attention
        x_encoding = cross_attention(norm_label, norm_img, norm_img) + label_encoding
        norm_x = x_layer_norm(x_encoding)  # Normalize cross attention output
        x_encoding = x_ff(norm_x) + x_encoding  # Add residual # Feed Forward Network
        
        # CROSS ATTENTION END!

        # PROJECT TO VOCAB
        logits = project_layer(x_encoding)
        # Reshape logits if needed
        if len(logits.shape) == 2:
            # logits shape: [sequence_length, vocab_size]
            # actual shape: [sequence_length]
            loss = loss_fn(logits, actual)
        else:
            # Ensure logits are 2D: [batch_size * sequence_length, vocab_size]
            batch_size = logits.size(0)
            seq_len = logits.size(1)
            logits = logits.view(-1, logits.size(-1))
            actual = actual.view(-1)
            loss = loss_fn(logits, actual)

        # Get predictions for each position in the sequence
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

        # Calculate accuracy
        correct = (predictions == actual).sum().item()
        total = len(actual)
        accuracy = correct / total

        # Update epoch metrics
        epoch_loss += loss.item()
        epoch_accuracy += accuracy
        num_batches += 1

        # BACKPROP
        optim.zero_grad()
        loss.backward()
        optim.step()

        wandb.log({
            "loss": loss.item(),
            "accuracy": accuracy
        })

        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{num_of_epochs}")
            print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")
            print(f"Predictions: {[id2label[p.item()] for p in predictions]}")
            print(f"Actual: {[id2label[a.item()] for a in actual]}")
            print("-" * 50)

    # Calculate epoch averages
    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_accuracy = epoch_accuracy / num_batches

    # Step the scheduler at the end of each epoch
    scheduler.step(avg_epoch_loss)

    # Log epoch metrics
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": avg_epoch_loss,
        "epoch_accuracy": avg_epoch_accuracy,
        "learning_rate": optim.param_groups[0]['lr']
    })

    # Print epoch summary
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Average Loss: {avg_epoch_loss:.4f}")
    print(f"Average Accuracy: {avg_epoch_accuracy:.2%}")
    print(f"Learning Rate: {optim.param_groups[0]['lr']:.6f}")
    print("=" * 50 + "\n")

wandb.finish()



