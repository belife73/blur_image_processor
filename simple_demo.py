#!/usr/bin/env python3
"""
简化演示脚本，展示模糊图像处理系统核心功能
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def main():
    print("=" * 60)
    print("BlurImageProcessor - 简化演示")
    print("=" * 60)
    
    # 导入核心模块
    from core.pipeline import BlurProcessor
    from utils.visualization import Visualizer
    
    print("\n1. 创建处理器...")
    processor = BlurProcessor()
    visualizer = Visualizer()
    
    print("\n2. 创建测试图像...")
    # 创建测试图像
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(test_image, "Test Image", (60, 128), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    cv2.rectangle(test_image, (50, 150), (200, 200), 200, 2)
    cv2.circle(test_image, (125, 175), 20, 150, -1)
    
    print(f"   测试图像创建完成，尺寸: {test_image.shape}")
    
    print("\n3. 添加模糊...")
    # 添加运动模糊
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, :] = 1.0 / kernel_size
    blurred_image = cv2.filter2D(test_image, -1, kernel)
    
    print(f"   模糊图像创建完成")
    
    print("\n4. 加载图像并检测模糊...")
    # 加载模糊图像
    processor.load_from_array(blurred_image)
    
    # 检测模糊
    is_blurry, score = processor.detect_blur(method='laplacian')
    print(f"   模糊检测结果: {'模糊' if is_blurry else '清晰'}")
    print(f"   模糊分数: {score:.2f}")
    
    print("\n5. 去模糊处理...")
    # 去模糊处理
    processor.deblur(method='wiener', balance=0.1)
    print("   维纳滤波去模糊完成")
    
    print("\n6. 图像增强...")
    # 图像增强
    processor.enhance(method='sharpening', amount=1.5)
    print("   锐化增强完成")
    
    print("\n7. 质量评估...")
    # 质量评估
    metrics = processor.evaluate_quality()
    print("   质量指标:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    print("\n8. 保存结果...")
    # 保存结果
    output_dir = current_dir / "data" / "demo_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_dir / "original.png"), test_image)
    cv2.imwrite(str(output_dir / "blurred.png"), blurred_image)
    processor.save_result(output_dir / "result.png")
    
    print(f"   结果保存到: {output_dir}")
    
    print("\n9. 可视化结果...")
    # 可视化结果
    try:
        images = [test_image, blurred_image, processor.processed_image]
        titles = ["原始图像", "模糊图像", "处理后图像"]
        
        visualizer.show_comparison(images, titles, show=False)
        
        # 保存可视化结果
        import matplotlib.pyplot as plt
        plt.savefig(str(output_dir / "comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   可视化结果保存到: {output_dir / 'comparison.png'}")
    except Exception as e:
        print(f"   可视化错误: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print(f"所有结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()