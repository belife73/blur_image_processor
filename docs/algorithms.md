# 算法文档

## 模糊检测算法

### 1. 拉普拉斯方差法 (Laplacian Variance)

**原理:**
清晰图像包含更多高频分量（边缘），模糊图像边缘平滑。通过计算拉普拉斯算子的方差来评估图像的锐利程度。

**公式:**
```
Score = Var(∇²I)
```

其中 `∇²I` 是图像的拉普拉斯算子，`Var` 表示方差。

**实现:**
```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
variance = laplacian.var()
is_blurry = variance < threshold
```

**特点:**
- 计算速度快
- 对散焦模糊效果好
- 阈值需要根据场景调整

---

### 2. 傅里叶变换分析 (FFT Analysis)

**原理:**
在频域中，清晰图像包含更多高频分量，模糊图像的高频分量衰减。通过分析频谱中的高频能量比例来判断模糊程度。

**步骤:**
1. 对图像进行二维傅里叶变换
2. 将零频分量移到频谱中心
3. 计算高频区域的能量占比
4. 与阈值比较

**实现:**
```python
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude = np.abs(fshift)
high_freq_energy = np.sum(magnitude[center_h-h//4:center_h+h//4, 
                                     center_w-w//4:center_w+w//4])
high_freq_ratio = high_freq_energy / np.sum(magnitude)
```

**特点:**
- 可以检测运动模糊方向
- 对噪声敏感
- 计算复杂度较高

---

### 3. 梯度分析法 (Gradient Analysis)

**原理:**
清晰图像的梯度变化剧烈，模糊图像的梯度变化平缓。通过计算图像梯度的方差来评估模糊程度。

**实现:**
```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_variance = gradient_magnitude.var()
```

**特点:**
- 对边缘敏感
- 可以结合边缘密度分析
- 计算效率高

---

## 去模糊算法

### 1. 维纳滤波 (Wiener Filter)

**原理:**
维纳滤波是一种经典的去卷积方法，在已知模糊核（PSF）和噪声统计特性的情况下，可以最小化均方误差。

**公式:**
```
G(u,v) = H*(u,v) / (|H(u,v)|² + SNR⁻¹)
F_est(u,v) = G(u,v) * F_blur(u,v)
```

其中 `H` 是模糊核的傅里叶变换，`SNR` 是信噪比。

**实现:**
```python
from skimage import restoration

deconvolved = restoration.wiener(img_float, psf, balance=0.1)
```

**特点:**
- 需要知道模糊核
- 对噪声有抑制作用
- 适用于轻微到中度模糊

---

### 2. Richardson-Lucy 算法

**原理:**
Richardson-Lucy 是一种迭代式去卷积算法，基于泊松噪声模型，适用于天文图像等低信噪比场景。

**迭代公式:**
```
f^(k+1) = f^(k) * (h ⊗ (g / (h ⊗ f^(k))))
```

其中 `f` 是估计的清晰图像，`h` 是模糊核，`g` 是观测图像。

**实现:**
```python
from skimage import restoration

deconvolved = restoration.richardson_lucy(img_float, psf, iterations=30)
```

**特点:**
- 迭代式算法
- 适用于低信噪比图像
- 可能产生振铃效应

---

### 3. 盲去卷积 (Blind Deconvolution)

**原理:**
在不知道模糊核的情况下，同时估计模糊核和清晰图像。

**方法:**
- 基于边缘预测的盲去卷积
- 基于稀疏性的盲去卷积
- 基于深度学习的盲去卷积

**实现:**
```python
from skimage import restoration

deconvolved, estimated_psf = restoration.unsupervised_wiener(
    img_float, initial_psf, reg=0.1, max_num_iter=50
)
```

**特点:**
- 不需要预先知道模糊核
- 计算复杂度高
- 结果可能不稳定

---

### 4. 反锐化掩模 (Unsharp Mask)

**原理:**
通过减去模糊版本来增强图像的高频分量。

**公式:**
```
I_sharp = I + α * (I - I_blurred)
```

其中 `α` 是增强强度。

**实现:**
```python
blurred = cv2.GaussianBlur(image, (0, 0), sigma)
sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
```

**特点:**
- 计算速度快
- 适用于轻度模糊
- 可能放大噪声

---

## 图像增强算法

### 1. 锐化处理

**方法:**
- 拉普拉斯锐化
- 反锐化掩模
- 高通滤波

**拉普拉斯锐化公式:**
```
I_sharp = I - λ * ∇²I
```

---

### 2. 对比度增强

**方法:**
- 直方图均衡化
- CLAHE (限制对比度自适应直方图均衡化)
- Gamma 校正
- 对比度拉伸

**CLAHE 优势:**
- 避免过度增强
- 自适应调整
- 保留局部细节

---

### 3. 降噪处理

**方法:**
- 高斯滤波
- 中值滤波
- 双边滤波
- 非局部均值去噪

**双边滤波特点:**
- 保留边缘
- 去除平滑区域噪声
- 计算复杂度较高

---

## 质量评估指标

### 1. PSNR (Peak Signal-to-Noise Ratio)

**公式:**
```
PSNR = 20 * log10(MAX_I / sqrt(MSE))
```

**特点:**
- 常用的图像质量指标
- 与主观感知不完全一致
- 对误差敏感

---

### 2. SSIM (Structural Similarity)

**公式:**
```
SSIM = (2μ_xμ_y + C₁)(2σ_xy + C₂) / (μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)
```

**特点:**
- 更符合人眼感知
- 考虑结构信息
- 范围在 [0, 1]

---

### 3. MSE (Mean Squared Error)

**公式:**
```
MSE = (1/MN) * ΣΣ (I(x,y) - K(x,y))²
```

**特点:**
- 简单直观
- 对大误差敏感
- 无量纲

---

## 模糊核估计

### 方法

1. **梯度法**: 基于图像梯度统计
2. **频谱法**: 基于傅里叶频谱分析
3. **边缘法**: 基于边缘特征分析
4. **深度学习法**: 使用神经网络估计

### 模糊类型分类

- **运动模糊**: 方向性模糊
- **散焦模糊**: 径向对称模糊
- **高斯模糊**: 各向同性模糊
