
# DDPM 实现对比：从零实现到使用扩散库

本项目包含三个不同实现层次的 DDPM（Denoising Diffusion Probabilistic Models）模型，用于在 MNIST 数据集上生成手写数字图像。三个文件分别展示了：

- 从零实现完整的 DDPM 训练与采样逻辑
- 使用 `diffusers` 库简化实现
- 一个极简版 DDPM 演示（不含时间步嵌入）

---

## 📁 文件说明

| 文件名 | 描述 |
|--------|------|
| `DDPM(real).py` | 完整实现 DDPM，包含线性 beta 调度、时间步嵌入、前向加噪、反向采样、损失曲线和生成过程可视化。 |
| `DDPM(use_pack).py` | 使用 `diffusers` 库的 `UNet2DModel` 和 `DDPMScheduler`，代码更简洁，适合快速原型开发。 |
| `DDPM.py` | 极简版 DDPM，不含时间步嵌入，使用简单的线性退化策略，适合理解扩散过程的基本思想。 |

---

## 🚀 运行方式

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- torchvision
- diffusers
- matplotlib
- tqdm（仅 `DDPM(use_pack).py` 使用）

你可以使用以下命令安装依赖：

```bash
pip install torch torchvision matplotlib diffusers tqdm
```

### 运行示例

```bash
# 运行完整版 DDPM（从零实现）
python DDPM\(real\).py

# 运行基于 diffusers 的版本
python DDPM\(use_pack\).py

# 运行极简版 DDPM
python DDPM.py
```

---

## 🧠 模型核心思想对比

| 特性 | `DDPM(real).py` | `DDPM(use_pack).py` | `DDPM.py` |
|------|----------------|----------------------|------------|
| 时间步嵌入 | ✅ 使用正弦位置编码 + MLP | ✅ 内置在 UNet 中 | ❌ 无 |
| 噪声调度 | 线性 beta 调度 | 余弦调度（更稳定） | 自定义线性退化 |
| 前向加噪 | 重参数化公式 | `scheduler.add_noise` | 简单加权混合 |
| 反向采样 | 手动实现采样公式 | `scheduler.step` | 线性去噪 |
| 训练目标 | 预测噪声 | 预测噪声 | 预测原图（非标准） |
| 损失曲线 | ✅ 按 epoch 记录 | ✅ 按 iteration 记录 | ✅ 按 epoch 记录 |
| 采样过程可视化 | ✅ 可选展示 | ✅ 自动展示 | ✅ 自动展示 |

> 注意：`DDPM.py` 的目标是预测原始图像而非噪声，这是一种简化版 DDPM，用于教学演示。

---

## 📊 输出示例

每个脚本运行后都会输出：

- 训练损失曲线（对数坐标）
- 最终生成的 8 张 MNIST 图像
- 采样过程的逐步可视化（从噪声到图像）

示例输出（`DDPM(real).py`）：

```
Using device: cuda
开始训练DDPM...
Epoch [1/10], Average Loss: 0.123456
...
开始生成图像...
采样进度: 0/1000
...
```

---

## 📌 注意事项

- `DDPM(real).py` 和 `DDPM(use_pack).py` 的训练时间较长（1000 个时间步），建议使用 GPU 运行。
- `DDPM.py` 训练速度快，适合快速理解扩散过程，但生成质量较低。
- 所有脚本都会自动下载 MNIST 数据集到 `data/` 目录。

---

## 📚 参考

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/index)

---

## 📄 作者
Eric
