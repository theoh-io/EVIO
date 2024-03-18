import numpy as np
import torch
from src.networks.fusion import LMF, Concat, EAF



cuda=False
# # Tensor Representation
# mod1=np.append(mod1, 1)
# mod2=np.append(mod2, 1)
# Z= np.outer(mod1,mod2)
# print("Z Matrix")
# print(Z.shape)
# h=np.random.rand(1,6)

# Basic concatenation

# Full rank Fusion
# dim_fr=[mod1.shape[1], mod2.shape[1], h.shape[1]]
# fr_W=np.random(dim_fr)
# result=np.dot(dim_fr, fr_W)

# create example of preprocesssed unimodal feature
batch_size=32
mod1_sz=128
mod2_sz=16
mod3_sz=56
input_dims=[mod1_sz, mod2_sz, mod3_sz]
output_dim=256

mod1=torch.tensor(np.random.rand(batch_size,mod1_sz)).float()
mod2=torch.tensor(np.random.rand(batch_size,mod2_sz)).float()
mod3=torch.tensor(np.random.rand(batch_size,mod3_sz)).float()

feat=torch.cat((mod1, mod2, mod3), dim=1)

dropout=[0, 0.1, 0.15, 0.2, 0.3, 0.5]
rank = [1, 4, 8, 16]

# Simple Concatenation Fusion
print('simple concat fusion')
model=Concat(input_dims, output_dim)
result= model(feat)
print(result.shape)

# Low Rank Fusion
print('low rank fusion')
for r in rank:
    model = LMF(input_dims, dropout, output_dim, r)
    if cuda:
        model = model.cuda()
        DTYPE = torch.cuda.FloatTensor
    print("Model initialized")
    result = model(feat)
    print(result.shape)

# # Attention-based Fusion
# print("External Attention Fusion")
# fusion_module=EAF(input_dims, output_dim)
# output_att=fusion_module(mod1.float(), mod2.float(), mod3.float())
# print(f"output attention {output_att}")
    