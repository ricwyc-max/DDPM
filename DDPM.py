__author__ = 'Eric'

#==================================1、导包===========================================
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler,UNet2DModel
from matplotlib import pyplot as plt
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device:{device}')
n_epochs = 15


#==================================2、载入数据集并展示===========================================
dataset = torchvision.datasets.MNIST(root='data/',train=True,download=True,transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(dataset,batch_size=8,shuffle=True)

x,y = next(iter(train_dataloader))
print('Input Shape:',x.shape)
print('Labels:',y)
plt.imshow(torchvision.utils.make_grid(x)[0],cmap='Greys')
plt.show()


#==================================3、构建一个网络模型能够接收噪声图像，并输出相同大小图片的预测结果,提供加噪函数===========================================
class BasicUNet(nn.Module):
    '''
    一个简单的Unet实现
    '''
    def __init__(self,in_channels=1,out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels,32,kernel_size = 5,padding = 2),# 多一个通道放时间
            nn.Conv2d(32,64,kernel_size = 5,padding = 2),
            nn.Conv2d(64,64,kernel_size = 5,padding = 2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64,64,kernel_size = 5,padding = 2),
            nn.Conv2d(64,32,kernel_size = 5,padding = 2),
            nn.Conv2d(32,out_channels,kernel_size = 5,padding = 2),
        ])
        self.act = nn.SiLU()#激活函数
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self,x):
        h = []
        #U-net
        for i,l in enumerate(self.down_layers):
            x = self.act(l(x))#遍历每一层，添加激活函数
            if i<2:#对于前2层
                h.append(x)#保存输出以供跳跃残差连接
                x = self.downscale(x)#下采样

        for i,l in enumerate(self.up_layers):
            if i>0:#对于第一层后面的层
                x = self.upscale(x)#先上采样
                x += h.pop()#弹出掐灭保存的输出进行残差连接
                #不添加激活函数保持残差连接的"恒等映射"特性
            x = self.act(l(x))#最后的一层添加激活函数

        return x



#退化过程，加噪
def corrupt(x,amount):
    """
    通过amount权重噪声对x进行加噪退化操作
    :param x:
    :param amount:
    :return:
    """
    noise = torch.rand_like(x)
    amount = amount.view(-1,1,1,1)#改变尺寸以供广播操作
    return x*(1-amount) + noise*amount

#对输出结果进行可视化，观察是否满足预期：
fig ,axs = plt.subplots(2,1,figsize=(12,5))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0],cmap="Greys")

amount = torch.linspace(0,1,x.shape[0])
noise_x = corrupt(x,amount)

axs[1].set_title('Corrrepted data (-- amount increase -->)')
axs[1].imshow(torchvision.utils.make_grid(noise_x)[0],cmap="Greys")

plt.show()



# 初始化模型、优化器和损失函数
net = BasicUNet(in_channels=1, out_channels=1).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # DDPM通常使用MSE损失

# 存储损失值
losses = []


#==================================4、训练模型===========================================
#训练循环
for epoch in range(n_epochs):
    epoch_loss = 0.0  # 记录当前epoch的总损失
    num_batches = 0   # 记录当前epoch的batch数量

    for x,y in train_dataloader:
        x = x.to(device)
        noise_amount = torch.rand(x.shape[0]).to(device)
        noise_x = corrupt(x,noise_amount)

        pred = net(noise_x)

        loss = loss_fn(pred,x)

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss+=loss.item()
        num_batches+=1

    # 每个epoch结束后打印一次平均损失
    avg_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.6f}')
    #保存损失为后续可视化
    losses.append(avg_loss)


# 训练完成后打印最终信息
print("训练完成！")

#==================================5、采样/推理过程===========================================
n_steps = 10
x = torch.rand(8, 1, 28, 28).to(device)  # 从随机开始
step_history = [x.detach().cpu()]

for i in range(n_steps):
    with torch.no_grad():
        # 计算当前步的噪声程度 (从高到低)
        current_amount = torch.ones(8).to(device) * (1 - i / n_steps)
        pred = net(x)
    mix_factor = 1 / (n_steps - i)
    x = x * (1 - mix_factor) + pred * mix_factor
    step_history.append(x.detach().cpu())

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

# 可视化采样过程
fig, axs = plt.subplots(n_steps + 1, 8, figsize=(12, (n_steps + 1) * 1.5))
for i in range(n_steps + 1):
    for j in range(8):
        axs[i, j].imshow(step_history[i][j, 0], cmap='Greys')
        axs[i, j].axis('off')
    axs[i, 0].set_ylabel(f'Step {i}', fontsize=12)
plt.suptitle('DDPM Sampling Process', fontsize=16)
plt.tight_layout()
plt.show()

# 可视化最终生成的图像
fig, axes = plt.subplots(1, 8, figsize=(12, 3))
for i in range(8):
    axes[i].imshow(step_history[-1][i, 0], cmap='Greys')
    axes[i].axis('off')
    axes[i].set_title(f'Sample {i+1}')
plt.suptitle('Generated MNIST Images', fontsize=14)
plt.show()




