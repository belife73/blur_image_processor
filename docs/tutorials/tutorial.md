# 教程

## 快速开始

### 安装

```bash
cd blur_image_processor
pip install -r requirements.txt
```

### 基础使用

```python
from src.core.pipeline import BlurProcessor

processor = BlurProcessor()
processor.load_image("path/to/image.jpg")

is_blurry, score = processor.detect_blur()

if is_blurry:
    processor.deblur(method='wiener')
    processor.enhance(method='sharpening')
    processor.save_result("output.jpg")
```

---

## 教程 1: 模糊检测

### 目标
学习如何使用不同的方法检测图像模糊。

### 步骤

1. **加载图像**
```python
from src.core.pipeline import BlurProcessor
from src.detection.blur_detector import BlurDetector

processor = BlurProcessor()
detector = BlurDetector()

processor.load_image("image.jpg")
```

2. **使用拉普拉斯方差法**
```python
is_blurry, score = detector.detect_laplacian(processor.gray_image)
print(f"模糊分数: {score:.2f}")
print(f"是否模糊: {is_blurry}")
```

3. **使用傅里叶变换分析**
```python
is_blurry, score = detector.detect_fft(processor.gray_image)
```

4. **使用梯度分析法**
```python
is_blurry, score = detector.detect_gradient(processor.gray_image)
```

5. **综合检测**
```python
results = detector.detect_all(processor.gray_image)
print(results)
```

---

## 教程 2: 去模糊处理

### 目标
学习如何使用不同的去模糊算法。

### 步骤

1. **维纳滤波**
```python
from src.deblur.deblur_engine import DeblurEngine

engine = DeblurEngine()

result = engine.deblur(
    processor.gray_image,
    method='wiener',
    balance=0.1
)
```

2. **Richardson-Lucy 算法**
```python
result = engine.deblur(
    processor.gray_image,
    method='richardson_lucy',
    iterations=30
)
```

3. **盲去卷积**
```python
result = engine.deblur(
    processor.gray_image,
    method='blind',
    iterations=50
)
```

4. **反锐化掩模**
```python
result = engine.deblur(
    processor.gray_image,
    method='unsharp',
    sigma=1.0,
    strength=1.5
)
```

---

## 教程 3: 图像增强

### 目标
学习如何增强图像质量。

### 步骤

1. **锐化处理**
```python
from src.enhancement.enhancer import ImageEnhancer

enhancer = ImageEnhancer()

result = enhancer.enhance(
    processor.gray_image,
    method='sharpening',
    amount=1.5
)
```

2. **对比度增强**
```python
result = enhancer.enhance(
    processor.gray_image,
    method='contrast',
    alpha=1.2,
    beta=10
)
```

3. **降噪处理**
```python
result = enhancer.enhance(
    processor.gray_image,
    method='denoise',
    method='bilateral'
)
```

---

## 教程 4: 质量评估

### 目标
学习如何评估图像质量。

### 步骤

1. **计算 PSNR**
```python
from src.analysis.quality_metrics import QualityMetrics

metrics = QualityMetrics()

psnr = metrics.calculate_psnr(image1, image2)
print(f"PSNR: {psnr:.2f} dB")
```

2. **计算 SSIM**
```python
ssim = metrics.calculate_ssim(image1, image2)
print(f"SSIM: {ssim:.4f}")
```

3. **计算所有指标**
```python
all_metrics = metrics.calculate_all(processed, original)
print(all_metrics)
```

---

## 教程 5: 可视化

### 目标
学习如何可视化处理结果。

### 步骤

1. **显示单张图像**
```python
from src.utils.visualization import Visualizer

visualizer = Visualizer()
visualizer.show_single(image, title="My Image")
```

2. **对比显示**
```python
images = [original, blurred, restored]
titles = ["Original", "Blurred", "Restored"]
visualizer.show_comparison(images, titles)
```

3. **显示直方图**
```python
visualizer.show_histogram(image)
```

4. **显示频谱**
```python
visualizer.show_fft(image)
```

5. **显示边缘检测**
```python
visualizer.show_edge_detection(image)
```

---

## 教程 6: 批量处理

### 目标
学习如何批量处理图像。

### 步骤

```python
from pathlib import Path
from src.core.pipeline import BlurProcessor

processor = BlurProcessor()
input_dir = Path("input_images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

for image_path in input_dir.glob("*.jpg"):
    processor.load_image(image_path)
    
    is_blurry, score = processor.detect_blur()
    
    if is_blurry:
        processor.deblur(method='wiener')
        processor.enhance(method='sharpening')
        
        output_path = output_dir / f"processed_{image_path.name}"
        processor.save_result(output_path)
    
    processor.reset()
```

---

## 教程 7: 自定义管道

### 目标
学习如何创建自定义处理管道。

### 步骤

```python
class CustomPipeline:
    def __init__(self):
        self.processor = BlurProcessor()
        self.detector = BlurDetector()
        self.engine = DeblurEngine()
        self.enhancer = ImageEnhancer()
    
    def process(self, image_path):
        self.processor.load_image(image_path)
        
        is_blurry, score = self.detector.detect_laplacian(
            self.processor.gray_image
        )
        
        if is_blurry:
            deblurred = self.engine.deblur(
                self.processor.gray_image,
                method='wiener'
            )
            
            enhanced = self.enhancer.enhance(
                deblurred,
                method='sharpening'
            )
            
            return enhanced
        
        return self.processor.gray_image
```

---

## 教程 8: 参数调优

### 目标
学习如何调整算法参数。

### 维纳滤波参数

- `balance`: 噪声平衡参数
  - 值越小，去模糊效果越强
  - 值越大，噪声抑制越强
  - 典型范围: 0.01 - 1.0

### Richardson-Lucy 参数

- `iterations`: 迭代次数
  - 值越大，效果越好但计算时间越长
  - 典型范围: 10 - 50

### 反锐化掩模参数

- `sigma`: 高斯模糊标准差
  - 值越大，增强范围越广
  - 典型范围: 0.5 - 3.0

- `strength`: 增强强度
  - 值越大，锐化效果越强
  - 典型范围: 0.5 - 2.0

---

## 教程 9: 处理不同类型的模糊

### 运动模糊

```python
from src.deblur.wiener_filter import WienerFilter

wiener = WienerFilter()
psf = wiener.create_motion_psf(length=15, angle=45)
result = wiener.deblur(image, psf=psf)
```

### 散焦模糊

```python
psf = wiener.create_gaussian_psf(size=7, sigma=1.0)
result = wiener.deblur(image, psf=psf)
```

---

## 教程 10: 性能优化

### 目标
学习如何提高处理速度。

### 技巧

1. **调整图像大小**
```python
from src.core.preprocessor import Preprocessor

preprocessor = Preprocessor()
resized = preprocessor.resize(image, scale=0.5)
```

2. **使用更快的算法**
```python
result = engine.deblur(image, method='unsharp')
```

3. **减少迭代次数**
```python
result = engine.deblur(
    image,
    method='richardson_lucy',
    iterations=10
)
```

4. **批量处理时使用多进程**
```python
from multiprocessing import Pool

def process_image(image_path):
    processor = BlurProcessor()
    processor.load_image(image_path)
    # ... 处理逻辑
    return result

with Pool(4) as p:
    results = p.map(process_image, image_paths)
```
