#!/usr/bin/env python3
"""
简单演示脚本，展示模糊图像处理系统功能
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def demo_blur_detection():
    print("=" * 60)
    print("演示 1: 模糊检测")
    print("=" * 60)
    
    from detection.blur_detector import BlurDetector
    
    # 创建测试图像
    sharp_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(sharp_image, "Sharp", (80, 128), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    # 创建模糊图像
    kernel = np.ones((10, 10)) / 100
    blurry_image = cv2.filter2D(sharp_image, -1, kernel)
    
    # 保存结果
    output_dir = current_dir / "data" / "demo_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_dir / "sharp_demo.png"), sharp_image)
    cv2.imwrite(str(output_dir / "blurry_demo.png"), blurry_image)
    
    # 检测模糊
    detector = BlurDetector()
    
    is_blurry_sharp, score_sharp = detector.detect_laplacian(sharp_image)
    is_blurry_blur, score_blur = detector.detect_laplacian(blurry_image)
    
    print(f"清晰图像 - 模糊: {is_blurry_sharp}, 分数: {score_sharp:.2f}")
    print(f"模糊图像 - 模糊: {is_blurry_blur}, 分数: {score_blur:.2f}")
    
    print(f"结果保存到: {output_dir}")


def demo_deblur():
    print("\n" + "=" * 60)
    print("演示 2: 去模糊处理")
    print("=" * 60)
    
    from deblur.deblur_engine import DeblurEngine
    
    # 加载模糊图像
    output_dir = current_dir / "data" / "demo_results"
    blurry_image = cv2.imread(str(output_dir / "blurry_demo.png"), cv2.IMREAD_GRAYSCALE)
    
    if blurry_image is None:
        print("错误: 无法加载模糊图像")
        return
    
    # 去模糊处理
    engine = DeblurEngine()
    
    # 维纳滤波
    wiener_result = engine.deblur(blurry_image, method='wiener', balance=0.1)
    cv2.imwrite(str(output_dir / "wiener_result.png"), wiener_result)
    print(f"维纳滤波完成，结果保存到: {output_dir / 'wiener_result.png'}")
    
    # Richardson-Lucy算法
    richardson_result = engine.deblur(blurry_image, method='richardson_lucy', iterations=30)
    cv2.imwrite(str(output_dir / "richardson_result.png"), richardson_result)
    print(f"Richardson-Lucy算法完成，结果保存到: {output_dir / 'richardson_result.png'}")
    
    # 反锐化掩模
    unsharp_result = engine.deblur(blurry_image, method='unsharp', sigma=1.0, strength=1.5)
    cv2.imwrite(str(output_dir / "unsharp_result.png"), unsharp_result)
    print(f"反锐化掩模完成，结果保存到: {output_dir / 'unsharp_result.png'}")


def demo_enhancement():
    print("\n" + "=" * 60)
    print("演示 3: 图像增强")
    print("=" * 60)
    
    from enhancement.enhancer import ImageEnhancer
    
    # 加载维纳滤波结果
    output_dir = current_dir / "data" / "demo_results"
    image = cv2.imread(str(output_dir / "wiener_result.png"), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("错误: 无法加载图像")
        return
    
    # 图像增强
    enhancer = ImageEnhancer()
    
    # 锐化处理
    sharpened = enhancer.enhance(image, method='sharpening', amount=1.5)
    cv2.imwrite(str(output_dir / "sharpened_result.png"), sharpened)
    print(f"锐化处理完成，结果保存到: {output_dir / 'sharpened_result.png'}")
    
    # 对比度增强
    contrast_enhanced = enhancer.enhance(image, method='contrast', alpha=1.2, beta=10)
    cv2.imwrite(str(output_dir / "contrast_result.png"), contrast_enhanced)
    print(f"对比度增强完成，结果保存到: {output_dir / 'contrast_result.png'}")
    
    # 降噪处理
    # 添加噪声
    noisy = image.copy()
    noise = np.random.normal(0, 25, noisy.shape).astype(np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    denoised = enhancer.enhance(noisy, method='denoise')
    cv2.imwrite(str(output_dir / "denoised_result.png"), denoised)
    print(f"降噪处理完成，结果保存到: {output_dir / 'denoised_result.png'}")


def demo_quality_metrics():
    print("\n" + "=" * 60)
    print("演示 4: 质量评估")
    print("=" * 60)
    
    from analysis.quality_metrics import QualityMetrics
    
    # 加载原始图像和处理结果
    output_dir = current_dir / "data" / "demo_results"
    original = cv2.imread(str(output_dir / "sharp_demo.png"), cv2.IMREAD_GRAYSCALE)
    processed = cv2.imread(str(output_dir / "sharpened_result.png"), cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None:
        print("错误: 无法加载图像")
        return
    
    # 计算质量指标
    metrics = QualityMetrics()
    
    psnr = metrics.calculate_psnr(processed, original)
    ssim = metrics.calculate_ssim(processed, original)
    mse = metrics.calculate_mse(processed, original)
    mae = metrics.calculate_mae(processed, original)
    
    print(f"PSNR (峰值信噪比): {psnr:.2f} dB")
    print(f"SSIM (结构相似性): {ssim:.4f}")
    print(f"MSE (均方误差): {mse:.2f}")
    print(f"MAE (平均绝对误差): {mae:.2f}")
    
    # 无参考质量指标
    no_ref_metrics = metrics.calculate_no_reference_metrics(processed)
    print("\n无参考质量指标:")
    for metric, value in no_ref_metrics.items():
        print(f"  {metric}: {value:.4f}")


def demo_pipeline():
    print("\n" + "=" * 60)
    print("演示 5: 完整处理管道")
    print("=" * 60)
    
    from core.pipeline import BlurProcessor
    
    # 创建处理器
    processor = BlurProcessor()
    
    # 创建测试图像
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(test_image, "Pipeline", (60, 128), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    
    # 添加模糊
    kernel = np.ones((8, 8)) / 64
    blurred_image = cv2.filter2D(test_image, -1, kernel)
    
    # 加载图像
    processor.load_from_array(blurred_image)
    
    # 检测模糊
    is_blurry, score = processor.detect_blur(method='laplacian')
    print(f"模糊检测结果: {is_blurry}, 分数: {score:.2f}")
    
    # 处理图像
    processor.deblur(method='wiener', balance=0.1)
    processor.enhance(method='sharpening', amount=1.2)
    
    # 评估质量
    metrics = processor.evaluate_quality()
    print("\n质量指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存结果
    output_dir = current_dir / "data" / "demo_results"
    processor.save_result(output_dir / "pipeline_result.png")
    print(f"\n结果保存到: {output_dir / 'pipeline_result.png'}")


def main():
    print("BlurImageProcessor - 功能演示")
    print("本演示将展示系统的各个功能模块")
    
    # 运行各个演示
    demo_blur_detection()
    demo_deblur()
    demo_enhancement()
    demo_quality_metrics()
    demo_pipeline()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("结果保存在: data/demo_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()