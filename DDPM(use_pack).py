__author__ = 'Eric'
"""
DDPM使用相关包
"""
#==================================1、导包===========================================
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 超参数
n_epochs = 100
batch_size = 64
learning_rate = 1e-3
image_size = 28
num_train_timesteps = 1000  # DDPM标准步数

#==================================2、载入数据集===========================================
dataset = torchvision.datasets.MNIST(
    root='data/',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])
)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 显示数据样本
x, y = next(iter(train_dataloader))
print('Input Shape:', x.shape)
plt.imshow(torchvision.utils.make_grid(x[:8])[0], cmap='Greys')
plt.title('Sample MNIST Images')
plt.show()

#==================================3、构建模型===========================================
# 使用diffusers提供的UNet2DModel
net = UNet2DModel(
    sample_size=image_size,      # 图像大小
    in_channels=1,                # 输入通道（灰度图）
    out_channels=1,               # 输出通道
    layers_per_block=2,           # 每个block的层数
    block_out_channels=(32, 64, 64),  # 各层通道数
    down_block_types=(
        "DownBlock2D",           # 下采样块
        "AttnDownBlock2D",       # 带注意力的下采样块
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
).to(device)

# 噪声调度器
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_schedule="squaredcos_cap_v2"  # 余弦调度（效果更好）
)

# 优化器
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

# 损失函数
loss_fn = nn.MSEloss()

print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

#==================================4、训练模型===========================================
losses = []

for epoch in range(n_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    # 使用tqdm显示进度
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
    
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        
        # 1. 随机采样时间步 t
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, 
            (batch_x.shape[0],), device=device
        ).long()
        
        # 2. 采样噪声
        noise = torch.randn_like(batch_x).to(device)
        
        # 3. 加噪（重参数化技巧）
        noisy_x = noise_scheduler.add_noise(batch_x, noise, timesteps)
        
        # 4. 模型预测噪声
        noise_pred = net(noisy_x, timesteps).sample
        
        # 5. 计算损失
        loss = loss_fn(noise_pred, noise)
        
        # 6. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        epoch_loss += loss.item()
        num_batches += 1
        losses.append(loss.item())
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.6f}')

print("训练完成！")

# 保存模型
torch.save(net.state_dict(), 'ddpm_mnist_diffusers.pth')
print("模型已保存")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

#==================================5、采样/推理过程===========================================
def sample_ddpm(model, noise_scheduler, n_samples=8, device='cuda'):
    """标准DDPM采样"""
    model.eval()
    
    # 从纯噪声开始
    x = torch.randn(n_samples, 1, image_size, image_size).to(device)
    
    # 存储生成过程（选择性存储）
    step_history = []
    key_steps = [0, 99, 199, 399, 599, 799, 999]
    
    with torch.no_grad():
        for t in range(noise_scheduler.num_train_timesteps - 1, -1, -1):
            timestep = torch.tensor([t] * n_samples, device=device).long()
            
            # 预测噪声
            noise_pred = model(x, timestep).sample
            
            # DDPM采样步骤
            x = noise_scheduler.step(noise_pred, t, x).prev_sample
            
            # 记录关键步骤
            if t in key_steps:
                step_history.append(x.detach().cpu())
    
    return x, step_history

# 生成图像
print("开始生成图像...")
generated_images, generation_steps = sample_ddpm(net, noise_scheduler, n_samples=8, device=device)

# 可视化最终生成的图像
fig, axes = plt.subplots(1, 8, figsize=(12, 3))
for i in range(8):
    # 反归一化到[0,1]
    img = (generated_images[i, 0] + 1) / 2
    axes[i].imshow(img, cmap='Greys')
    axes[i].axis('off')
    axes[i].set_title(f'Sample {i+1}')
plt.suptitle('Generated MNIST Images (DDPM with diffusers)', fontsize=14)
plt.show()

# 可视化生成过程
fig, axs = plt.subplots(len(generation_steps), 8, figsize=(12, len(generation_steps) * 1.5))
for i, step_imgs in enumerate(generation_steps):
    for j in range(8):
        img = (step_imgs[j, 0] + 1) / 2
        axs[i, j].imshow(img, cmap='Greys')
        axs[i, j].axis('off')
    axs[i, 0].set_ylabel(f'Step {[0, 99, 199, 399, 599, 799, 999][i]}', fontsize=12)
plt.suptitle('DDPM Sampling Process', fontsize=16)
plt.tight_layout()
plt.show()

#==================================6、与真实图像对比===========================================
fig, axes = plt.subplots(2, 8, figsize=(12, 6))

# 第一行：真实图像
real_images, _ = next(iter(train_dataloader))
for i in range(8):
    axes[0, i].imshow(real_images[i, 0], cmap='Greys')
    axes[0, i].axis('off')
axes[0, 0].set_ylabel('Real', fontsize=12)

# 第二行：生成图像
for i in range(8):
    img = (generated_images[i, 0] + 1) / 2
    axes[1, i].imshow(img, cmap='Greys')
    axes[1, i].axis('off')
axes[1, 0].set_ylabel('Generated', fontsize=12)

plt.suptitle('Real vs Generated MNIST', fontsize=14)
plt.tight_layout()
plt.show()