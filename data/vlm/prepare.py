# data_prep_docvqa.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from datasets import load_dataset
from PIL import Image
import tiktoken

# --- config ---
IMG_SIZE = 100
CTX_LEN = 1024
enc = tiktoken.get_encoding("gpt2")

# --- preprocessors ---

def pad_to_square(img: Image.Image, size=IMG_SIZE):
    """resize keeping aspect and center pad to (size, size)."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    pad_w, pad_h = size - new_w, size - new_h
    pad_left, pad_top = pad_w // 2, pad_h // 2
    pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top
    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), 0, 'constant')

img_transform = transforms.Compose([
    pad_to_square,
    transforms.ToTensor(),  # [0,1]
])

# --- dataset ---

class VisionQADataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = load_dataset("moondream/vision-ai-checkup", split=split)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # image
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        
        # Always convert to RGB to ensure 3 channels
        img = img.convert("RGB")
        img = img_transform(img)

        # text
        text = f"Q: {sample['prompt']} A: {sample['answer']}"
        ids = enc.encode_ordinary(text)[:CTX_LEN-1]
        ids.append(enc.eot_token)
        ids = torch.tensor(ids, dtype=torch.long)

        return {"image": img, "tokens": ids, "len": len(ids)}

# --- collate ---

def collate_fn(batch):
    max_len = max(x["len"] for x in batch)
    B = len(batch)
    toks = torch.full((B, max_len), fill_value=enc.eot_token, dtype=torch.long)
    imgs = torch.stack([x["image"] for x in batch])
    for i, x in enumerate(batch):
        toks[i, :x["len"]] = x["tokens"]
    return {"images": imgs, "tokens": toks}

# --- usage ---

if __name__ == "__main__":
    dataset = VisionQADataset(split='train')
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    for batch in loader:
        print(batch["images"].shape, batch["tokens"].shape)
        break
