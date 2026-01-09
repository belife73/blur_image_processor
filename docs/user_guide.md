# BlurImageProcessor 用户指南

## 1. 项目概述

BlurImageProcessor 是一个功能强大的基于传统算法的模糊图像处理与分析系统，提供了从模糊检测、去模糊到图像增强和质量评估的完整解决方案。

### 1.1 核心功能

- **模糊检测**：多种算法检测图像模糊程度
- **去模糊处理**：多种去卷积算法恢复清晰图像
- **图像增强**：锐化、对比度调整、降噪等
- **质量评估**：多种客观质量指标
- **可视化**：丰富的图像分析和结果展示
- **GUI界面**：直观易用的图形用户界面

### 1.2 应用场景

- 监控摄像头图像增强
- 文档扫描图像清晰化
- 老旧照片修复
- 医学图像去模糊
- 遥感图像增强
- 摄影图像后期处理

## 2. 安装指南

### 2.1 前提条件

- Python 3.9+
- pip 包管理器
- 操作系统：Windows、macOS 或 Linux

### 2.2 安装步骤

1. **进入项目目录**
   ```bash
   cd /root/Image Restoration/blur_image_processor
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **可选：使用 setup.py 安装**
   ```bash
   pip install -e .
   ```

## 3. 快速开始

### 3.1 使用 GUI 界面（推荐）

如果你的环境有图形显示，推荐使用 GUI 界面，这是最直观的使用方式：

```bash
python gui.py
```

**GUI界面主要功能**：
- 左侧显示原始图像，右侧显示处理后图像
- 顶部控制栏：加载图像、保存结果、重置
- 功能标签页：模糊检测、去模糊、图像增强
- 底部质量评估面板

### 3.2 使用命令行脚本

**运行演示脚本**：
```bash
python demo.py
```

这将运行一个完整的功能演示，结果保存到 `data/demo_results/` 目录。

**运行示例代码**：
```bash
# 基础使用示例
python examples/basic_usage.py

# 批量处理示例
python examples/batch_processing.py

# 自定义管道示例
python examples/custom_pipeline.py
```

### 3.3 使用 Python API

```python
from src.core.pipeline import BlurProcessor

# 创建处理器
processor = BlurProcessor()

# 加载图像
processor.load_image("path/to/image.jpg")

# 检测模糊
is_blurry, score = processor.detect_blur(method='laplacian')
print(f"模糊检测结果: {'模糊' if is_blurry else '清晰'}, 分数: {score:.2f}")

# 如果模糊，进行去模糊处理
if is_blurry:
    # 去模糊
    processor.deblur(method='wiener', balance=0.1)
    
    # 图像增强
    processor.enhance(method='sharpening', amount=1.5)
    
    # 评估质量
    metrics = processor.evaluate_quality()
    print(f"PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}")
    
    # 保存结果
    processor.save_result("output.jpg")
```

## 4. 核心功能详细介绍

### 4.1 模糊检测

BlurImageProcessor 提供了多种模糊检测算法：

| 算法 | 原理 | 特点 |
|------|------|------|
| 拉普拉斯方差法 | 基于图像边缘锐利度 | 计算速度快，适合实时检测 |
| FFT 分析 | 频域分析 | 能检测出模糊类型，计算量大 |
| 梯度分析法 | 图像梯度统计 | 对不同模糊类型鲁棒性好 |
| 综合检测 | 结合多种方法 | 准确率最高，计算量较大 |

**使用示例**：

```python
from src.detection.blur_detector import BlurDetector

# 创建检测器
detector = BlurDetector()

# 检测模糊
is_blurry, score = detector.detect_laplacian(image)
print(f"模糊检测结果: {'模糊' if is_blurry else '清晰'}, 分数: {score:.2f}")
```

### 4.2 去模糊处理

提供多种去模糊算法：

| 算法 | 原理 | 适用场景 |
|------|------|----------|
| 维纳滤波 | 最小均方误差估计 | 已知模糊类型，计算速度快 |
| Richardson-Lucy | 迭代去卷积 | 复杂模糊，泊松噪声模型 |
| 盲去卷积 | 自动估计模糊核 | 未知模糊类型 |
| 反锐化掩模 | 增强高频分量 | 轻微模糊，快速处理 |

**使用示例**：

```python
from src.deblur.deblur_engine import DeblurEngine

# 创建去模糊引擎
engine = DeblurEngine()

# 使用维纳滤波去模糊
result = engine.deblur(image, method='wiener', balance=0.1)

# 使用 Richardson-Lucy 去模糊
result = engine.deblur(image, method='richardson_lucy', num_iter=30)
```

### 4.3 图像增强

提供多种图像增强方法：

| 方法 | 效果 | 适用场景 |
|------|------|----------|
| 锐化 | 增强边缘和细节 | 轻微模糊，细节丢失 |
| 对比度调整 | 提高图像对比度 | 对比度低，图像暗淡 |
| 降噪 | 减少图像噪声 | 噪声明显，影响视觉效果 |

**使用示例**：

```python
from src.enhancement.enhancer import ImageEnhancer

# 创建增强器
enhancer = ImageEnhancer()

# 锐化处理
sharpened = enhancer.enhance(image, method='sharpening', amount=1.5)

# 对比度增强
contrast = enhancer.enhance(image, method='contrast', alpha=1.2, beta=10)

# 降噪处理
denoised = enhancer.enhance(image, method='denoise')
```

### 4.4 质量评估

提供多种客观质量指标：

| 指标 | 描述 | 范围 | 越高越好 |
|------|------|------|----------|
| PSNR | 峰值信噪比 | 0-∞ | 是 |
| SSIM | 结构相似性 | 0-1 | 是 |
| MSE | 均方误差 | 0-∞ | 否 |
| MAE | 平均绝对误差 | 0-∞ | 否 |

**使用示例**：

```python
from src.analysis.quality_metrics import QualityMetrics

# 创建质量评估器
metrics = QualityMetrics()

# 计算 PSNR
psnr = metrics.calculate_psnr(processed_image, original_image)

# 计算 SSIM
ssim = metrics.calculate_ssim(processed_image, original_image)

# 计算所有指标
all_metrics = metrics.calculate_all(processed_image, original_image)
```

## 5. GUI 界面详细使用

### 5.1 启动界面

运行 `python gui.py` 后，你将看到如下界面：

- **顶部控制栏**：包含加载图像、保存结果、重置按钮
- **图像显示区域**：左侧原始图像，右侧处理后图像
- **功能标签页**：三个标签页分别对应模糊检测、去模糊、图像增强
- **质量评估面板**：底部显示质量指标

### 5.2 基本操作流程

1. **加载图像**
   - 点击 "加载图像" 按钮
   - 选择要处理的图像文件（支持 JPG、PNG、BMP、TIFF 格式）
   - 图像将显示在左侧面板

2. **检测模糊**
   - 切换到 "模糊检测" 标签页
   - 选择检测方法（默认为拉普拉斯方差法）
   - 点击 "检测模糊" 按钮
   - 查看检测结果

3. **去模糊处理**
   - 切换到 "去模糊" 标签页
   - 选择去模糊方法
   - 调整相应参数
   - 点击 "执行去模糊" 按钮
   - 查看右侧面板的处理结果

4. **图像增强**
   - 切换到 "图像增强" 标签页
   - 选择增强方法
   - 调整相应参数
   - 点击 "执行增强" 按钮
   - 查看右侧面板的处理结果

5. **评估质量**
   - 点击 "评估质量" 按钮
   - 查看底部面板的质量指标

6. **保存结果**
   - 点击 "保存结果" 按钮
   - 选择保存路径和格式
   - 点击 "保存"

### 5.3 参数调整技巧

- **维纳滤波**：平衡参数越小，去模糊效果越强，但可能引入噪声
- **Richardson-Lucy**：迭代次数越多，效果越好，但计算时间越长
- **锐化**：锐化强度适中，过高会引入噪声
- **对比度**：对比度调整要适度，避免过曝或过暗

## 6. 高级使用

### 6.1 批量处理

使用示例脚本 `examples/batch_processing.py` 可以批量处理多张图像：

```bash
python examples/batch_processing.py --input_dir input_images --output_dir output_images --method wiener
```

### 6.2 自定义处理管道

你可以创建自定义的处理管道：

```python
from src.core.pipeline import BlurProcessor

# 创建处理器
processor = BlurProcessor()

# 加载图像
processor.load_image("path/to/image.jpg")

# 自定义处理流程
processor.detect_blur(method='all')
processor.deblur(method='richardson_lucy', num_iter=50)
processor.enhance(method='sharpening', amount=1.0)
processor.enhance(method='contrast', alpha=1.1)
processor.enhance(method='denoise')

# 评估质量
metrics = processor.evaluate_quality()

# 保存结果
processor.save_result("custom_pipeline_result.jpg")
```

### 6.3 配置文件

项目使用 `config.yaml` 管理默认参数，你可以根据需要修改：

```yaml
detection:
  laplacian:
    threshold: 100.0  # 模糊阈值，可根据需要调整
    kernel_size: 3
    
deblur:
  wiener:
    balance: 0.1      # 维纳滤波平衡参数
    psf_size: 5
    
enhancement:
  sharpening:
    kernel_size: 3
    amount: 1.5       # 锐化强度
```

## 7. 示例教程

### 7.1 示例 1：修复模糊的文档照片

**问题**：扫描的文档照片模糊，文字不清晰

**解决方案**：

1. 加载文档照片
2. 使用拉普拉斯方差法检测模糊（分数：45.2，模糊）
3. 使用 Richardson-Lucy 算法去模糊（迭代次数：30）
4. 使用锐化增强细节（强度：1.2）
5. 评估质量：PSNR=28.56, SSIM=0.9723
6. 保存结果

**效果**：文字清晰可见，可读性大大提高

### 7.2 示例 2：增强监控摄像头图像

**问题**：监控摄像头拍摄的夜间图像模糊，噪点多

**解决方案**：

1. 加载监控图像
2. 使用 FFT 分析检测模糊（分数：72.5，模糊）
3. 使用维纳滤波去模糊（平衡参数：0.05）
4. 使用降噪处理减少噪点
5. 使用对比度增强提高亮度（alpha=1.3）
6. 评估质量：PSNR=24.32, SSIM=0.9456
7. 保存结果

**效果**：图像更清晰，噪点减少，细节可见

## 8. 常见问题解答

### 8.1 GUI 界面无法启动

**问题**：运行 `python gui.py` 后报错，提示 "no display name and no $DISPLAY environment variable"

**解决方案**：
- 确保你在有图形显示的环境中运行
- 对于远程服务器，需要配置 X11 转发
- 或者使用命令行脚本和 Python API

### 8.2 处理速度慢

**解决方案**：
- 对于大图像，先调整尺寸
- 选择计算速度快的算法（如维纳滤波、拉普拉斯方差法）
- 减少迭代次数（对于迭代算法）

### 8.3 去模糊效果不理想

**解决方案**：
- 尝试不同的去模糊算法
- 调整算法参数
- 结合图像增强（锐化、对比度调整）
- 对于严重模糊，可能需要专业的深度学习方法

### 8.4 质量指标如何解读

- **PSNR**：一般来说，PSNR > 30 dB 表示质量良好
- **SSIM**：SSIM > 0.9 表示质量良好
- **MSE**：数值越小越好
- **MAE**：数值越小越好

## 9. 故障排除

### 9.1 依赖安装问题

如果安装依赖时出错，可以尝试：

```bash
# 升级 pip
pip install --upgrade pip

# 单独安装有问题的包
pip install opencv-python numpy scikit-image matplotlib

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 9.2 图像格式问题

如果无法加载某些图像格式，可以尝试：

```bash
# 安装额外的图像处理库
pip install imageio

# 或转换为常用格式
```

### 9.3 内存不足

处理大图像时可能会遇到内存不足问题：

```python
# 调整图像尺寸
import cv2
image = cv2.imread("large_image.jpg")
resized = cv2.resize(image, (800, 600))  # 调整为合适尺寸
```

## 10. 扩展和定制

### 10.1 添加新的算法

1. 在相应模块目录下创建新的算法文件
2. 实现算法类和方法
3. 在 `__init__.py` 中导出新算法
4. 更新配置文件

### 10.2 定制 GUI 界面

你可以修改 `gui.py` 文件来定制 GUI 界面：
- 添加新的功能按钮
- 调整布局和样式
- 添加新的算法参数控制

### 10.3 集成到其他项目

```python
# 作为库导入
from blur_image_processor.src.core.pipeline import BlurProcessor

# 或添加到系统路径
sys.path.append("path/to/blur_image_processor")
```

## 11. 技术支持

如果遇到问题或有建议，欢迎通过以下方式联系：

- 项目地址: [https://github.com/yourusername/blur-image-processor](https://github.com/yourusername/blur-image-processor)
- 邮箱: your.email@example.com

## 12. 更新日志

### v1.0.0 (2025-12-26)
- 初始版本发布
- 实现了完整的模糊图像处理功能
- 包含多种经典算法的实现
- 提供了详细的文档和示例
- 新增GUI界面

## 13. 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 14. 致谢

感谢所有为这个项目做出贡献的开发者和研究人员！

## 15. 引用

如果您在研究中使用了本项目，请考虑引用：

```
@software{BlurImageProcessor,
  title = {BlurImageProcessor: A Comprehensive Blur Image Processing System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/blur-image-processor},
}
```

---

**祝您使用愉快！**

如果本指南对您有帮助，请给项目一个 Star 支持！
