import torch 
import torch.nn as nn 
from torch.nn import functional as F
import random

#Reading text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

#HyperPrams 
batch_size = 32
block_size = 128 
max_iters = 5000
learning_rate = 0.001
eval_iters = 1000
n_embd = 16
n_head = 4
n_layer = 4
dropout = 0.2


chars = sorted(list(set(text)))
vocab_size = len(chars)

#Creating mapping for characters to int 
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x]) 

#Data loading
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y 

xb, yb = get_batch('train')

train_data[:block_size + 1]

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#self attention 

class Head(nn.Module):

    def __init__(self, C, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out 
    
#multihead attention
class MultiHead(nn.Module):

    def __init__(self, C, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(C, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
#feed forward network

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  
            nn.Linear(4 * n_embd, n_embd), #projection layer
            nn.Dropout(dropout),  

        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHead(n_embd, head_size, 4)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa_head(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x
    
#Bigram model 
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_linear = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.positional_embedding(torch.arange(T)).unsqueeze(0).expand(B, -1, -1)       
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)
        return logits, loss 
    
    def generate(self, idx, nmax): 
        for _ in range(nmax):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)
            idx = torch.cat([idx, idx_next.unsqueeze(-1)], dim=-1)
        return idx


model = BigramLanguageModel()


print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), nmax=100)[0].tolist()))


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


#training loop
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step{iter}: train loss is {losses['train']}, val_loss is {losses['val']}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward() 
    optimizer.step()
 

print(loss.item())
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), nmax=100)[0].tolist()))
