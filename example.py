import random
import torch
from PIL import Image
from torchvision import datasets, transforms

class Combine(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.tf = transforms.ToTensor()
        self.ds = datasets.MNIST(root='.', train=True, transform=self.tf, download=True)
        # self.ln = len(self.ds)

    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, idx):
        idx = random.sample(range(len(self)), 4)
        store = []
        label = []

        for i in idx:
            x, y = self.ds[i]
            store.append(x)
            label.append(y)

        img = Image.new('L', (56, 56))
        img.paste(store[0], (0, 0))
        img.paste(store[1], (28, 0))
        img.paste(store[2], (0, 28))
        img.paste(store[3], (28, 28))

        return img, label

# Instantiate the custom dataset and fetch a sample
ds = Combine()
img, label = ds[0]
print(label)
img.show()
