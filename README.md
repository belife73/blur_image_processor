# BlurImageProcessor

一个功能强大的基于传统算法的模糊图像处理与分析系统。

## 项目概述

BlurImageProcessor 是一个全面的图像处理框架，专注于模糊图像的检测、去模糊处理、增强和质量评估。它提供了多种经典算法的实现，适用于各种模糊图像处理场景。

## 核心功能

### 1. 模糊检测
- **拉普拉斯方差法** - 基于图像边缘锐利度检测模糊
- **傅里叶变换分析** - 通过频域分析评估模糊程度
- **梯度分析法** - 利用图像梯度统计特征检测模糊
- **综合检测** - 结合多种方法进行更准确的模糊判断

### 2. 去模糊处理
- **维纳滤波** - 经典的去卷积方法，适用于已知模糊核的场景
- **Richardson-Lucy 算法** - 基于泊松噪声模型的迭代式去卷积
- **盲去卷积** - 自动估计模糊核的去模糊方法
- **反锐化掩模** - 快速增强图像边缘的方法

### 3. 图像增强
- **锐化处理** - 多种锐化算法，包括拉普拉斯锐化和反锐化掩模
- **对比度增强** - 直方图均衡化、CLAHE和Gamma校正
- **降噪处理** - 高斯滤波、中值滤波、双边滤波和非局部均值去噪

### 4. 质量评估
- **PSNR** (峰值信噪比) - 衡量像素误差
- **SSIM** (结构相似性) - 衡量视觉感知的相似度
- **MSE** (均方误差) - 衡量平均像素差异
- **MAE** (平均绝对误差) - 衡量平均绝对像素差异
- **无参考指标** - 包括锐度、对比度、亮度和熵

### 5. 可视化与GUI
- 单张图像显示
- 多张图像对比
- 直方图显示
- 傅里叶频谱分析
- 边缘检测结果可视化
- 质量指标图表
- **GUI界面** - 直观易用的图形用户界面，支持图像加载、实时处理和结果保存

## 安装

### 前提条件
- Python 3.9+
- pip 包管理器

### 安装步骤

1. 克隆或下载项目：
   ```bash
   cd /path/to/your/workspace
   ```

2. 安装依赖：
   ```bash
   cd blur_image_processor
   pip install -r requirements.txt
   ```

3. 或者使用 setup.py 安装：
   ```bash
   pip install -e .
   ```

## 快速开始

### 基础使用

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

### 运行演示

```bash
python demo.py
```

这将运行一个完整的功能演示，包括模糊检测、去模糊、图像增强和质量评估，并将结果保存到 `data/demo_results/` 目录。

### 运行示例代码

```bash
# 基础使用示例
python examples/basic_usage.py

# 批量处理示例
python examples/batch_processing.py

# 自定义管道示例
python examples/custom_pipeline.py
```

### 运行GUI界面

```bash
python gui.py
```

这将启动一个直观易用的图形用户界面，支持：
- 图像加载和实时预览
- 多种模糊检测算法
- 多种去模糊和图像增强方法
- 实时质量评估
- 结果保存

## 核心模块

### 1. BlurProcessor

主要的处理管道类，提供完整的图像处理流程：
- 图像加载和预处理
- 模糊检测
- 去模糊处理
- 图像增强
- 质量评估
- 结果保存

### 2. BlurDetector

模糊检测器，提供多种检测方法：
- `detect_laplacian()` - 使用拉普拉斯方差法
- `detect_fft()` - 使用傅里叶变换分析
- `detect_gradient()` - 使用梯度分析法
- `detect_all()` - 综合多种方法

### 3. DeblurEngine

去模糊引擎，封装了多种去模糊算法：
- `deblur(method='wiener')` - 维纳滤波
- `deblur(method='richardson_lucy')` - Richardson-Lucy算法
- `deblur(method='blind')` - 盲去卷积
- `deblur(method='unsharp')` - 反锐化掩模

### 4. ImageEnhancer

图像增强器，提供多种增强方法：
- `enhance(method='sharpening')` - 锐化处理
- `enhance(method='contrast')` - 对比度增强
- `enhance(method='denoise')` - 降噪处理

### 5. QualityMetrics

质量评估器，计算各种图像质量指标：
- `calculate_psnr()` - 计算PSNR
- `calculate_ssim()` - 计算SSIM
- `calculate_mse()` - 计算MSE
- `calculate_all()` - 计算所有指标
- `calculate_no_reference_metrics()` - 计算无参考指标

## 项目结构

```
blur_image_processor/
├── data/                 # 数据目录
│   ├── demo_results/     # 演示结果
│   ├── processed_images/ # 批量处理结果
│   └── sample_images/    # 示例图像
├── docs/                 # 文档
│   ├── algorithms.md     # 算法说明
│   ├── api_reference.md  # API参考
│   └── tutorials/        # 教程
├── examples/             # 示例代码
│   ├── basic_usage.py    # 基础使用示例
│   ├── batch_processing.py # 批量处理示例
│   └── custom_pipeline.py # 自定义管道示例
├── src/                  # 源代码
│   ├── analysis/         # 分析模块
│   ├── core/             # 核心模块
│   ├── deblur/           # 去模糊模块
│   ├── detection/        # 模糊检测模块
│   ├── enhancement/      # 图像增强模块
│   └── utils/            # 工具模块
├── tests/                # 测试文件
│   ├── test_deblur.py     # 去模糊测试
│   ├── test_detection.py  # 模糊检测测试
│   └── test_enhancement.py # 图像增强测试
├── config.yaml           # 配置文件
├── demo.py               # 演示脚本
├── gui.py                # GUI界面
├── requirements.txt      # 依赖列表
├── setup.py              # 安装配置
└── README.md             # 项目说明
```

## API 参考

### BlurProcessor

| 方法 | 描述 |
|------|------|
| `load_image(path, color_mode='BGR')` | 加载图像文件 |
| `load_from_array(array)` | 从numpy数组加载图像 |
| `detect_blur(method='laplacian', **kwargs)` | 检测图像模糊程度 |
| `deblur(method='wiener', **kwargs)` | 执行去模糊处理 |
| `enhance(method='sharpening', **kwargs)` | 执行图像增强 |
| `evaluate_quality(reference=None)` | 计算质量指标 |
| `save_result(path, image=None, quality=95)` | 保存结果图像 |
| `process(detect_method='laplacian', deblur_method='wiener', enhance_method='sharpening', **kwargs)` | 执行完整处理流程 |
| `visualize_results(show=True)` | 可视化处理结果 |

### BlurDetector

| 方法 | 描述 |
|------|------|
| `detect_laplacian(image, threshold=None)` | 使用拉普拉斯方差法检测模糊 |
| `detect_fft(image, threshold=None)` | 使用傅里叶变换分析检测模糊 |
| `detect_gradient(image, threshold=None)` | 使用梯度分析法检测模糊 |
| `detect_all(image, threshold=None)` | 综合多种方法检测模糊 |
| `get_blur_level(image)` | 获取模糊级别 |

### DeblurEngine

| 方法 | 描述 |
|------|------|
| `deblur(image, method='wiener', **kwargs)` | 执行去模糊处理 |

### QualityMetrics

| 方法 | 描述 |
|------|------|
| `calculate_psnr(image1, image2)` | 计算峰值信噪比 |
| `calculate_ssim(image1, image2)` | 计算结构相似性 |
| `calculate_mse(image1, image2)` | 计算均方误差 |
| `calculate_mae(image1, image2)` | 计算平均绝对误差 |
| `calculate_all(image1, image2=None)` | 计算所有质量指标 |
| `calculate_no_reference_metrics(image)` | 计算无参考质量指标 |

## 配置文件

项目使用 `config.yaml` 配置文件来管理默认参数：

```yaml
detection:
  laplacian:
    threshold: 100.0
    kernel_size: 3
    
deblur:
  wiener:
    balance: 0.1
    psf_size: 5
    
enhancement:
  sharpening:
    kernel_size: 3
    amount: 1.5
```

## 示例输出

### 模糊检测结果
```
模糊检测结果: 模糊, 分数: 65.23
```

### 质量评估指标
```
PSNR: 25.78 dB
SSIM: 0.9634
MSE: 171.96
MAE: 3.39
```

## 技术栈

- **编程语言**: Python 3.9+
- **核心库**: OpenCV, NumPy, Scikit-image, Matplotlib, SciPy
- **GUI框架**: Tkinter (Python标准库)
- **图像处理**: Pillow
- **配置管理**: PyYAML
- **进度显示**: tqdm

## 应用场景

- 监控摄像头图像增强
- 文档扫描图像清晰化
- 老旧照片修复
- 医学图像去模糊
- 遥感图像增强
- 摄影图像后期处理

## 性能优化

1. **图像尺寸调整**: 对于大图像，可以先调整尺寸以提高处理速度
2. **算法选择**: 根据具体场景选择合适的算法
3. **参数调优**: 根据图像特性调整算法参数
4. **并行处理**: 批量处理时可以考虑使用多进程

## 贡献指南

欢迎贡献代码和改进建议！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目地址: [https://github.com/yourusername/blur-image-processor](https://github.com/yourusername/blur-image-processor)
- 邮箱: your.email@example.com

## 更新日志

### v1.0.0 (2025-12-26)
- 初始版本发布
- 实现了完整的模糊图像处理功能
- 包含多种经典算法的实现
- 提供了详细的文档和示例
- **新增GUI界面** - 直观易用的图形用户界面，支持图像加载、实时处理和结果保存

## 致谢

感谢所有为这个项目做出贡献的开发者和研究人员！

## 引用

如果您在研究中使用了本项目，请考虑引用：

```
@software{BlurImageProcessor,
  title = {BlurImageProcessor: A Comprehensive Blur Image Processing System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/blur-image-processor},
}
```
