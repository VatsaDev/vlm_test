# train_vlm.py
"""
same CLI / usage as original train.py. this file integrates:
- VisionProj (patch linear proj -> n_embd)
- VLM wrapper (image tokens + text tokens)
- HF dataset loader for moondream/vision-ai-checkup with center-pad to 100x100
- dataloader loop (replaces memmap get_batch)
"""

import os, time, math, pickle, inspect
from contextlib import nullcontext
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torchvision import transforms
from torchvision.transforms import functional as TF

from datasets import load_dataset
import tiktoken

# import your GPT/GPTConfig class from model.py
from model import GPTConfig, GPT

# ---------------- config (same vars as original train.py) ----------------
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'gpt2'  # 'scratch'|'resume'|'gpt2'
wandb_log = False
wandb_project = 'vlm'
wandb_run_name = 'vlm'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# vision / dataset settings
IMG_SIZE = 100        # final resolution
PATCH_SIZE = 10       # patch size; results in (10x10) patches -> 100 tokens if no cls
USE_CLS = True
DATASET_ID = "moondream/vision-ai-checkup"

# ---------------- command-line overrides if you have configurator.py -------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ---------------- ddp / device setup (same as original) ---------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

# ---------------- vision proj + vlm wrapper ---------------------------------
class VisionProj(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, n_embd=n_embd, use_cls_token=USE_CLS):
        super().__init__()
        self.p = patch_size
        self.proj = nn.Linear(3 * patch_size * patch_size, n_embd)
        self.use_cls = use_cls_token
        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x: (B,3,H,W) float tensor in [0,1]
        B, C, H, W = x.shape
        p = self.p
        h_p = H // p
        w_p = W // p
        if h_p == 0 or w_p == 0:
            raise ValueError("image smaller than patch size")
        x = x[:, :, :h_p*p, :w_p*p]
        x = x.unfold(2, p, p).unfold(3, p, p)   # (B,3,h_p,w_p,p,p)
        x = x.contiguous().permute(0,2,3,1,4,5) # (B,h_p,w_p,3,p,p)
        x = x.reshape(B, h_p*w_p, 3*p*p)
        x = self.proj(x)
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        return self.ln(x)

class VLM(nn.Module):
    def __init__(self, gpt: GPT, vision_proj: VisionProj, image_prefix_limit=None):
        super().__init__()
        self.gpt = gpt
        self.vision = vision_proj
        assert gpt.config.n_embd == vision_proj.proj.out_features
        self.image_prefix_limit = image_prefix_limit

    def forward(self, images, idx, targets=None):
        device = idx.device
        B = idx.size(0)
        txt_emb = self.gpt.transformer.wte(idx)          # (B, T_txt, n_embd)
        if images is None:
            return self.gpt(idx, targets)               # fallback to text-only routine

        img_emb = self.vision(images.to(device))        # (B, N_img, n_embd)
        if self.image_prefix_limit is not None and img_emb.size(1) > self.image_prefix_limit:
            img_emb = img_emb[:, :self.image_prefix_limit, :]

        x = torch.cat([img_emb, txt_emb], dim=1)        # (B, N_img + T_txt, n_embd)
        total_len = x.size(1)
        if total_len > self.gpt.config.block_size:
            # trunc text tail to fit
            keep = self.gpt.config.block_size - img_emb.size(1)
            assert keep > 0, "image tokens exceed block size"
            x = torch.cat([img_emb, txt_emb[:, -keep:, :]], dim=1)
            total_len = x.size(1)

        pos = torch.arange(0, total_len, dtype=torch.long, device=device)
        pos_emb = self.gpt.transformer.wpe(pos).unsqueeze(0)
        x = self.gpt.transformer.drop(x + pos_emb)

        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        logits = self.gpt.lm_head(x)

        if targets is not None:
            pad = torch.full((B, img_emb.size(1)), -100, dtype=torch.long, device=device)
            labels = torch.cat([pad, targets.to(device)], dim=1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            return logits, loss
        else:
            return logits, None

    @torch.no_grad()
    def generate(self, images, idx, max_new_tokens, temperature=1.0, top_k=None):
        device = idx.device
        img_emb = None if images is None else self.vision(images.to(device))
        for _ in range(max_new_tokens):
            if img_emb is None:
                logits, _ = self.gpt(idx)
                logits = logits[:, -1, :] / temperature
            else:
                txt_emb = self.gpt.transformer.wte(idx)
                x = torch.cat([img_emb, txt_emb], dim=1)
                total_len = x.size(1)
                if total_len > self.gpt.config.block_size:
                    need = self.gpt.config.block_size - img_emb.size(1)
                    txt_emb = txt_emb[:, -need:, :]
                    x = torch.cat([img_emb, txt_emb], dim=1)
                pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
                pos_emb = self.gpt.transformer.wpe(pos).unsqueeze(0)
                x = self.gpt.transformer.drop(x + pos_emb)
                for block in self.gpt.transformer.h:
                    x = block(x)
                x = self.gpt.transformer.ln_f(x)
                logits = self.gpt.lm_head(x)[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ---------------- dataset / dataloader (replacement for memmap) -------------
def pad_to_square_center(img: Image.Image, size=IMG_SIZE):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w*scale), int(h*scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w//2, pad_h//2
    right, bottom = pad_w - left, pad_h - top
    return TF.pad(img, (left, top, right, bottom), 0, 'constant')

to_tensor = transforms.ToTensor()  # [0,1]

class VisionQADataset(Dataset):
    def __init__(self, split='train'):
        self.ds = load_dataset(DATASET_ID, split=split)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        img = row['image']
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')
        img = pad_to_square_center(img, IMG_SIZE)
        img = to_tensor(img)
        text = f"Q: {row['prompt']} A: {row['answer']}"
        ids = enc.encode_ordinary(text)[:block_size-1]
        ids.append(EOT)
        ids = torch.tensor(ids, dtype=torch.long)
        return {'image': img, 'tokens': ids, 'len': ids.numel()}

def collate_fn(batch):
    B = len(batch)
    max_len = max(x['len'] for x in batch)
    toks = torch.full((B, max_len), fill_value=EOT, dtype=torch.long)
    imgs = torch.stack([x['image'] for x in batch])
    for i,x in enumerate(batch):
        toks[i, :x['len']] = x['tokens']
    return {'images': imgs, 'tokens': toks}

# ---------------- model init (mostly copied from original) -------------------
# model args and init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    model_args['vocab_size'] = 50304 if not os.path.exists('data/meta.pkl') else pickle.load(open('data/meta.pkl','rb'))['vocab_size']
    gptconf = GPTConfig(**model_args)
    gpt = GPT(gptconf, ["MLP"] * n_layer)
elif init_from == 'resume':
    ckpt = torch.load(os.path.join(out_dir,'ckpt.pt'), map_location='cpu')
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
        model_args[k] = ckpt['model_args'][k]
    gptconf = GPTConfig(**model_args)
    gpt = GPT(gptconf)
    gpt.load_state_dict(ckpt['model'])
elif init_from.startswith('gpt2'):
    gpt = GPT.from_pretrained(init_from, override_args=dict(dropout=dropout))
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
        model_args[k] = getattr(gpt.config,k)
else:
    raise ValueError("invalid init_from")

# patch in vision proj + wrap
vision = VisionProj(patch_size=PATCH_SIZE, n_embd=gpt.config.n_embd, use_cls_token=USE_CLS)
vlm = VLM(gpt, vision, image_prefix_limit=None)

# optional partial freeze - keep off by default; toggle as needed
PARTIAL_FREEZE = True
if PARTIAL_FREEZE:
    for name, p in gpt.named_parameters():
        # unfreeze final two blocks and ln_f
        if not any(k in name for k in ['ln_f', f'transformer.h.{gpt.config.n_layer-1}', f'transformer.h.{gpt.config.n_layer-2}']):
            p.requires_grad = False

vlm.to(device)

# wrap with torch.compile if desired (compile the raw module before DDP)
raw_model = vlm
if compile:
    print("compiling model...")
    raw_model = torch.compile(raw_model)

if ddp:
    raw_model = DDP(raw_model, device_ids=[ddp_local_rank])

model = raw_model

# ---------------- optimizer (build param groups from vlm.named_parameters) ----------
param_dict = {pn: p for pn,p in vlm.named_parameters()}
param_dict = {n:p for n,p in param_dict.items() if p.requires_grad}
decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0},
]
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)

# load optimizer if resume
if init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer']); del ckpt

# scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# ---------------- data loaders -----------------------------------------------
train_ds = VisionQADataset(split='train')
val_ds = VisionQADataset(split='train') if False else VisionQADataset(split='train')  # replace with validation split if available

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)
train_iter = iter(train_loader)

# helper to fetch next batch (loops over epoch automatically)
def get_next_batch():
    global train_iter
    try:
        b = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        b = next(train_iter)
    imgs = b['images'].to(device, non_blocking=True)
    toks = b['tokens'].to(device, non_blocking=True)
    # shift for language modelling targets (we keep full toks; VLM masks image tokens)
    return imgs, toks, toks

# ---------------- estimate_loss (keeps similar behavior) ----------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    vlm.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Ximg, Xtok, Y = get_next_batch()
            with ctx:
                _, loss = vlm(Ximg, Xtok, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    vlm.train()
    return out

# ---------------- lr schedule (same) ----------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ---------------- training loop (adapted) -----------------------------------
iter_num = 0
best_val_loss = 1e9
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                ckpt = {
                    'model': vlm.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # gradient accumulation loop
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            Ximg, Xtok, Y = get_next_batch()
            logits, loss = vlm(Ximg, Xtok, Y)
            loss = loss / gradient_accumulation_steps
        Ximg, Xtok, Y = get_next_batch()  # prefetch next
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time(); dt = t1 - t0; t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            try:
                mfu = vlm.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            except Exception:
                mfu = 0.0
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()