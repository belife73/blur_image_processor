#!/usr/bin/env python3
"""
简单的测试脚本，验证模糊图像处理系统
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

def test_basic_functionality():
    print("=" * 60)
    print("BlurImageProcessor - 基础功能测试")
    print("=" * 60)
    
    try:
        # 测试导入
        print("\n1. 测试模块导入...")
        from core.pipeline import BlurProcessor
        from detection.blur_detector import BlurDetector
        from deblur.deblur_engine import DeblurEngine
        from enhancement.enhancer import ImageEnhancer
        from analysis.quality_metrics import QualityMetrics
        from utils.visualization import Visualizer
        print("   所有模块导入成功!")
        
        # 创建测试图像
        print("\n2. 创建测试图像...")
        test_image = np.zeros((256, 256), dtype=np.uint8)
        cv2.putText(test_image, "Test", (80, 128), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.rectangle(test_image, (50, 150), (200, 200), 200, 2)
        cv2.circle(test_image, (125, 175), 20, 150, -1)
        
        # 添加模糊
        kernel = np.ones((10, 10)) / 100
        blurred_image = cv2.filter2D(test_image, -1, kernel)
        
        print(f"   测试图像创建完成，尺寸: {test_image.shape}")
        
        # 测试模糊检测
        print("\n3. 测试模糊检测...")
        detector = BlurDetector()
        
        is_blurry_sharp, score_sharp = detector.detect_laplacian(test_image)
        is_blurry_blur, score_blur = detector.detect_laplacian(blurred_image)
        
        print(f"   清晰图像 - 模糊: {is_blurry_sharp}, 分数: {score_sharp:.2f}")
        print(f"   模糊图像 - 模糊: {is_blurry_blur}, 分数: {score_blur:.2f}")
        
        # 测试去模糊
        print("\n4. 测试去模糊...")
        engine = DeblurEngine()
        
        deblurred = engine.deblur(blurred_image, method='wiener', balance=0.1)
        print(f"   去模糊完成，结果尺寸: {deblurred.shape}")
        
        # 测试图像增强
        print("\n5. 测试图像增强...")
        enhancer = ImageEnhancer()
        
        enhanced = enhancer.enhance(deblurred, method='sharpening', amount=1.5)
        print(f"   图像增强完成，结果尺寸: {enhanced.shape}")
        
        # 测试质量评估
        print("\n6. 测试质量评估...")
        metrics = QualityMetrics()
        
        psnr = metrics.calculate_psnr(enhanced, test_image)
        ssim = metrics.calculate_ssim(enhanced, test_image)
        mse = metrics.calculate_mse(enhanced, test_image)
        
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim:.4f}")
        print(f"   MSE: {mse:.2f}")
        
        # 保存结果
        print("\n7. 保存结果...")
        output_dir = current_dir / "data" / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_dir / "original.png"), test_image)
        cv2.imwrite(str(output_dir / "blurred.png"), blurred_image)
        cv2.imwrite(str(output_dir / "deblurred.png"), deblurred)
        cv2.imwrite(str(output_dir / "enhanced.png"), enhanced)
        
        print(f"   结果保存到: {output_dir}")
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    print("\n" + "=" * 60)
    print("BlurImageProcessor - 管道测试")
    print("=" * 60)
    
    try:
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
        print("\n1. 加载图像...")
        processor.load_from_array(blurred_image)
        
        # 检测模糊
        print("\n2. 检测模糊...")
        is_blurry, score = processor.detect_blur(method='laplacian')
        print(f"   模糊检测结果: {is_blurry}, 分数: {score:.2f}")
        
        # 去模糊处理
        print("\n3. 去模糊处理...")
        # 强制处理以测试管道
        processor.deblur(method='wiener', balance=0.1)
        processor.enhance(method='sharpening', amount=1.2)
        print("   处理完成")
        
        # 评估质量
        print("\n4. 评估质量...")
        metrics = processor.evaluate_quality()
        print("   质量指标:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value:.4f}")
        
        # 保存结果
        print("\n5. 保存结果...")
        output_dir = Path(__file__).parent / "data" / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processor.save_result(output_dir / "pipeline_result.png")
        print(f"   结果保存到: {output_dir / 'pipeline_result.png'}")
        
        print("\n" + "=" * 60)
        print("管道测试通过!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行基础功能测试
    basic_success = test_basic_functionality()
    
    # 运行管道测试
    pipeline_success = test_pipeline()
    
    if basic_success and pipeline_success:
        print("\n🎉 所有测试通过! 项目运行正常!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败!")
        sys.exit(1)