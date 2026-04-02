__author__ = 'Eric'
"""
DDPM原文复现
"""
#==================================1、导包===========================================
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device:{device}')
n_epochs = 10
timesteps = 1000

#==================================2、载入数据集并展示===========================================
dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

x, y = next(iter(train_dataloader))
print('Input Shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
plt.show()

#==================================3、预计算噪声调度系数===========================================
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    线性beta调度
    :param timesteps: 总时间步数
    :param start: 起始beta值
    :param end: 结束beta值
    :return: beta序列
    """
    return torch.linspace(start, end, timesteps)

# 计算所有需要的系数
betas = linear_beta_schedule(timesteps)                    # beta_t
alphas = 1 - betas                                          # alpha_t = 1 - beta_t
alphas_cumprod = torch.cumprod(alphas, dim=0)              # alpha_bar_t = prod(alpha_1..alpha_t)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # alpha_bar_{t-1}

# 重参数化需要的系数
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)           # sqrt(alpha_bar_t)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)  # sqrt(1 - alpha_bar_t)

# 后验分布计算需要的系数
posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # 后验方差
sqrt_recip_alphas = torch.sqrt(1 / alphas)                  # 1/sqrt(alpha_t)

# 将系数移动到设备上
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
alphas_cumprod_prev = alphas_cumprod_prev.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
posterior_variance = posterior_variance.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)

#==================================4、构建网络模型（加入时间步嵌入）===========================================
class TimeEmbedding(nn.Module):
    """
    时间步嵌入层，将时间步t转换为特征向量
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()
    
    def forward(self, t):
        # 正弦位置编码
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 通过MLP
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)
        return emb

class BasicUNet(nn.Module):
    """
    带时间步条件的U-Net模型
    """
    def __init__(self, in_channels=1, out_channels=1, time_dim=32):
        super().__init__()
        
        # 时间步嵌入
        self.time_mlp = TimeEmbedding(time_dim)
        
        # 下采样层
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        ])
        
        # 上采样层
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        ])
        
        self.time_cond_layers = nn.ModuleList([
            nn.Linear(time_dim, 32),  # 对应 down_layers[0]
            nn.Linear(time_dim, 64),  # 对应 down_layers[1]
            nn.Linear(time_dim, 64),  # 对应 down_layers[2]
            nn.Linear(time_dim, 64),  # 对应 up_layers[0]
            nn.Linear(time_dim, 32),  # 对应 up_layers[1]
            nn.Linear(time_dim, 1),   # 对应 up_layers[2]（输出层）
        ])
        
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x, t):
        """
        :param x: 噪声图像 [batch, channels, height, width]
        :param t: 时间步 [batch]
        :return: 预测的噪声
        """
        # 获取时间步嵌入
        t_emb = self.time_mlp(t)  # [batch, time_dim]
        
        h = []
        
        # 下采样路径（使用索引 0,1,2）
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))

            # 注入时间步条件 - 使用索引 i (0,1,2)
            time_cond = self.time_cond_layers[i](t_emb)  # 改这里
            time_cond = time_cond[:, :, None, None]
            x = x + time_cond

            if i < len(self.down_layers) - 1:
                h.append(x)
                x = self.downscale(x)

        # 上采样路径（使用索引 3,4,5）
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x = x + h.pop()

            x = self.act(layer(x))

            # 注入时间步条件 - 使用索引 i+3 (3,4,5)
            time_cond = self.time_cond_layers[i + 3](t_emb)  # 改这里
            time_cond = time_cond[:, :, None, None]
            x = x + time_cond

        return x

# 初始化模型、优化器和损失函数
net = BasicUNet(in_channels=1, out_channels=1).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # DDPM使用MSE损失预测噪声

# 存储损失值
losses = []

#==================================5、训练模型（真正的DDPM训练逻辑）===========================================
print("开始训练DDPM...")

for epoch in range(n_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for x, y in train_dataloader:
        x = x.to(device)
        batch_size = x.shape[0]
        
        # 1. 随机采样时间步 t（对于batch中的每个样本）
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        
        # 2. 生成随机噪声
        noise = torch.randn_like(x)
        
        # 3. 使用重参数化技巧一步加噪到时间步t
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noise_x = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise
        
        # 4. 模型预测噪声
        pred_noise = net(noise_x, t)
        
        # 5. 计算损失（预测噪声 vs 真实噪声）
        loss = loss_fn(pred_noise, noise)
        
        # 6. 反向传播
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # 每个epoch结束后打印平均损失
    avg_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.6f}')
    losses.append(avg_loss)

print("训练完成！")

#==================================6、采样/推理过程（真正的DDPM采样）===========================================
@torch.no_grad()
def sample_ddpm(n_samples, device, img_size=28, channels=1):
    """
    DDPM采样过程，从纯噪声逐步去噪生成图像
    :param n_samples: 生成样本数量
    :param device: 设备
    :param img_size: 图像尺寸
    :param channels: 通道数
    :return: 生成的图像
    """
    # 从纯噪声开始
    x = torch.randn(n_samples, channels, img_size, img_size).to(device)
    
    # 逐步去噪（从时间步T到1）
    for t_step in reversed(range(timesteps)):
        # 为batch中的每个样本创建时间步张量
        t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
        
        # 预测当前时间步的噪声
        pred_noise = net(x, t)
        
        # 计算均值（使用DDPM采样公式）
        # mu = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * pred_noise)
        beta_t = betas[t].view(-1, 1, 1, 1)
        alpha_t = alphas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # 计算预测的均值
        pred_mean = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_alpha_cumprod_t)
        
        # 添加噪声（除了最后一步）
        if t_step > 0:
            posterior_var_t = posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            x = pred_mean + torch.sqrt(posterior_var_t) * noise
        else:
            x = pred_mean
        
        # 可选：每100步打印一次进度
        if t_step % 200 == 0:
            print(f'采样进度: {timesteps - t_step}/{timesteps}')
    
    return x

# 生成图像
print("开始生成图像...")
n_samples = 8
generated_images = sample_ddpm(n_samples, device)

# 将生成的图像转换到[0,1]范围用于显示
generated_images = (generated_images + 1) / 2  # 如果模型输出范围是[-1,1]
generated_images = torch.clamp(generated_images, 0, 1)

#==================================7、可视化结果===========================================
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('DDPM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# 可视化生成的图像
fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
for i in range(n_samples):
    axes[i].imshow(generated_images[i, 0].cpu(), cmap='Greys')
    axes[i].axis('off')
    axes[i].set_title(f'Sample {i+1}')
plt.suptitle('DDPM Generated MNIST Images', fontsize=14)
plt.tight_layout()
plt.show()

# 可选：显示采样过程的中间步骤（如果想看逐步去噪过程）
def show_sampling_process(n_samples=4, n_steps_to_show=5):
    """
    展示采样过程的中间步骤
    """
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    steps_to_show = np.linspace(0, timesteps-1, n_steps_to_show, dtype=int)
    step_images = []
    
    for t_step in reversed(range(timesteps)):
        t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
        pred_noise = net(x, t)
        
        beta_t = betas[t].view(-1, 1, 1, 1)
        alpha_t = alphas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        pred_mean = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t_step > 0:
            posterior_var_t = posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            x = pred_mean + torch.sqrt(posterior_var_t) * noise
        else:
            x = pred_mean
        
        if t_step in steps_to_show:
            step_images.append(x.clone())
    
    # 可视化
    fig, axes = plt.subplots(len(step_images), n_samples, figsize=(n_samples * 2, len(step_images) * 2))
    for i, img in enumerate(step_images):
        for j in range(n_samples):
            axes[i, j].imshow(img[j, 0].cpu(), cmap='Greys')
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'Step {steps_to_show[i]}', fontsize=10)
    plt.suptitle('DDPM Sampling Process (noise -> image)', fontsize=14)
    plt.tight_layout()
    plt.show()

# 可选：取消注释以显示采样过程
show_sampling_process(n_samples=4, n_steps_to_show=6)