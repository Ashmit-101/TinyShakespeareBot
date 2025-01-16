import torch
import torch.nn as nn
import torch.nn.functional as F

# version 4: self-attention!
torch.manual_seed (1337)
B, T, C = 4,8,32 # batch, time, channels
x = torch. randn (B, T,C)
# let's see a single Head perform self-attention
head_size = 16
key = nn. Linear(C, head_size, bias=False)
query = nn. Linear(C, head_size, bias=False)
value = nn. Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query (x) # (B, T, 16)
wei = q @ k. transpose (-2, -1) # (В, T, 16) @ (В, 16, T) --> (В, T, T)
tril = torch. tril(torch.ones (T, T))
#wei = torch. zeros ((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))  #deoding block -- nodes can talk to each other if needed (for sentiment analysis)
wei = F.softmax(wei, dim=-1)
v = value (x)
out = wei @ v
#out = wei @ x

print(wei[0])
print(out. shape)


'''
cross attention: when you have another set of nodes that you want to attend to. 
ie instead of key(x) =====> key(x, y) where y is another set of nodes.

'''