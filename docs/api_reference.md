# API 参考文档

## BlurProcessor

主要的处理管道类，提供完整的图像处理流程。

### 初始化

```python
from src.core.pipeline import BlurProcessor

processor = BlurProcessor(config_path="config.yaml")
```

### 方法

#### `load_image(path, color_mode='BGR')`
加载图像文件。

**参数:**
- `path` (str/Path): 图像文件路径
- `color_mode` (str): 颜色模式，可选 'BGR', 'RGB', 'GRAY'

**返回:** np.ndarray

#### `detect_blur(method='laplacian', **kwargs)`
检测图像模糊程度。

**参数:**
- `method` (str): 检测方法，可选 'laplacian', 'fft', 'gradient', 'all'

**返回:** (is_blurry, score) 或 results 字典

#### `deblur(method='wiener', **kwargs)`
执行去模糊处理。

**参数:**
- `method` (str): 去模糊方法，可选 'wiener', 'richardson_lucy', 'blind', 'unsharp'

**返回:** np.ndarray

#### `enhance(method='sharpening', **kwargs)`
执行图像增强。

**参数:**
- `method` (str): 增强方法，可选 'sharpening', 'contrast', 'denoise'

**返回:** np.ndarray

#### `evaluate_quality(reference=None)`
计算图像质量指标。

**参数:**
- `reference` (np.ndarray): 参考图像，默认使用原始图像

**返回:** dict - 包含 PSNR, SSIM, MSE, MAE 等指标

#### `process(detect_method='laplacian', deblur_method='wiener', enhance_method='sharpening', auto_deblur=True, **kwargs)`
执行完整的处理流程。

**返回:** dict - 包含所有处理结果和指标

---

## BlurDetector

模糊检测器，提供多种检测方法。

### 方法

#### `detect_laplacian(image, threshold=None)`
使用拉普拉斯方差法检测模糊。

#### `detect_fft(image, threshold=None)`
使用傅里叶变换分析检测模糊。

#### `detect_gradient(image, threshold=None)`
使用梯度分析法检测模糊。

#### `detect_all(image, threshold=None)`
使用所有方法检测模糊并返回综合结果。

---

## DeblurEngine

去模糊引擎，提供多种去模糊算法。

### 方法

#### `deblur(image, method='wiener', **kwargs)`
执行去模糊处理。

**可用方法:**
- `wiener`: 维纳滤波
- `richardson_lucy`: Richardson-Lucy 算法
- `blind`: 盲去卷积
- `unsharp`: 反锐化掩模

---

## ImageEnhancer

图像增强器，提供多种增强方法。

### 方法

#### `enhance(image, method='sharpening', **kwargs)`
执行图像增强。

**可用方法:**
- `sharpening`: 锐化处理
- `contrast`: 对比度增强
- `denoise`: 降噪处理

---

## QualityMetrics

图像质量评估器。

### 方法

#### `calculate_psnr(image1, image2)`
计算峰值信噪比。

#### `calculate_ssim(image1, image2)`
计算结构相似性。

#### `calculate_mse(image1, image2)`
计算均方误差。

#### `calculate_all(image1, image2=None)`
计算所有质量指标。

---

## Visualizer

可视化工具类。

### 方法

#### `show_single(image, title="Image", cmap='gray', show=True)`
显示单张图像。

#### `show_comparison(images, titles, cmap='gray', show=True)`
显示多张图像对比。

#### `show_histogram(image, title="Histogram", show=True)`
显示图像直方图。

#### `show_fft(image, title="FFT Spectrum", show=True)`
显示傅里叶频谱。

#### `show_edge_detection(image, title="Edge Detection", show=True)`
显示边缘检测结果。

---

## ImageLoader

图像加载器。

### 方法

#### `load(path, color_mode='BGR')`
加载图像文件。

#### `save(image, path, quality=95)`
保存图像文件。

#### `get_image_info(image=None)`
获取图像信息。

#### `load_batch(directory, pattern='*')`
批量加载图像。

---

## Preprocessor

图像预处理器。

### 方法

#### `to_grayscale(image)`
转换为灰度图像。

#### `normalize(image, target_range=(0.0, 1.0))`
归一化图像。

#### `resize(image, size=None, scale=None, keep_aspect=True)`
调整图像大小。

#### `crop_center(image, crop_size)`
从中心裁剪图像。

#### `pad_image(image, padding, mode='constant', value=0)`
填充图像。

#### `adjust_brightness(image, factor=1.0)`
调整亮度。

#### `adjust_contrast(image, factor=1.0)`
调整对比度。

#### `gamma_correction(image, gamma=1.0)`
Gamma 校正。

#### `histogram_equalization(image, clip_limit=None)`
直方图均衡化。
