'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-09-20 23:05:40
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-09-20 23:10:32
FilePath: /schen/workspace/WatChMaL/test.py
Description: 
'''
import torch, time, os, math
from torch.amp import autocast, GradScaler

device = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ==== 构建与你训练完全一致的模型 ====
# model = instantiate(config.model).to(device)  # 如果你有 hydra
from watchmal.model.SwinTransformernewp2DI import SwinRegressorDI  # 仅示例
model = SwinRegressorDI(
   img_size=  (192,192),
in_chans_main=  2,
in_chans_mpmt=  38,
pre_stage_embed_dim= 48,
pre_stage_depth= 8,
pre_stage_num_heads= 6,
pre_stage_window_size= 4  ,
xattn_first_k_layers= 2 ,
xattn_num_heads= 6,
embed_dim= 96,
depths= [8,8,6,2],
num_heads= [6,8,12,24],
window_size= 6,
mlp_ratio= 4.0,
drop_path_rate= 0,

).to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

B = 32  # 用你训练时的每卡 batch
x_main = torch.randn(B, 2, 192, 192, device=device)
x_mpmt = torch.randn(B, 38, 192, 192, device=device)

def step():
    opt.zero_grad(set_to_none=True)
    with autocast("cuda"):
        out = model(x_main, x_mpmt)
        loss = out.float().pow(2).mean()
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()

# Warmup
for _ in range(30):
    step(); 
torch.cuda.synchronize()

# Measure
N = 10
t0 = time.time()
for _ in range(N):
    step()
torch.cuda.synchronize()
dt = time.time()-t0
print(f"Throughput: {B*N/dt:.2f} samples/s, step_time ~ {dt/N*1000:.1f} ms")
