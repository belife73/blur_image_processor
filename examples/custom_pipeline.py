import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.pipeline import BlurProcessor
from detection.blur_detector import BlurDetector
from deblur.deblur_engine import DeblurEngine
from enhancement.enhancer import ImageEnhancer
from analysis.quality_metrics import QualityMetrics
from utils.visualization import Visualizer


class CustomPipeline:
    def __init__(self):
        self.processor = BlurProcessor()
        self.detector = BlurDetector()
        self.deblur_engine = DeblurEngine()
        self.enhancer = ImageEnhancer()
        self.metrics = QualityMetrics()
        self.visualizer = Visualizer()
    
    def process_with_custom_steps(self, image_path, steps):
        print(f"\n加载图像: {image_path}")
        self.processor.load_image(image_path)
        
        current_image = self.processor.gray_image.copy()
        results = {
            'images': [current_image.copy()],
            'titles': ['原始图像'],
            'metrics': []
        }
        
        for step in steps:
            step_name = step.get('name', 'unknown')
            method = step.get('method')
            params = step.get('params', {})
            
            print(f"执行步骤: {step_name} (方法: {method})")
            
            if step_name == 'detect_blur':
                is_blurry, score = self.detector.detect_laplacian(current_image, **params)
                print(f"  模糊检测结果: {score:.2f}")
                results['blur_score'] = score
            
            elif step_name == 'deblur':
                current_image = self.deblur_engine.deblur(current_image, method=method, **params)
                results['images'].append(current_image.copy())
                results['titles'].append(f"去模糊 ({method})")
            
            elif step_name == 'enhance':
                current_image = self.enhancer.enhance(current_image, method=method, **params)
                results['images'].append(current_image.copy())
                results['titles'].append(f"增强 ({method})")
            
            elif step_name == 'evaluate':
                if len(results['images']) > 1:
                    metrics = self.metrics.calculate_all(
                        results['images'][-1],
                        results['images'][0]
                    )
                    results['metrics'].append(metrics)
                    print(f"  质量指标: {metrics}")
        
        return results
    
    def compare_methods(self, image_path, methods):
        print(f"\n比较不同去模糊方法: {image_path}")
        self.processor.load_image(image_path)
        
        results = {
            'images': [self.processor.gray_image.copy()],
            'titles': ['原始图像']
        }
        
        for method_name, method_params in methods.items():
            print(f"应用方法: {method_name}")
            
            deblurred = self.deblur_engine.deblur(
                self.processor.gray_image.copy(),
                method=method_name,
                **method_params
            )
            
            results['images'].append(deblurred)
            results['titles'].append(method_name)
            
            metrics = self.metrics.calculate_all(deblurred, self.processor.gray_image)
            print(f"  PSNR: {metrics.get('PSNR', 0):.2f}, SSIM: {metrics.get('SSIM', 0):.4f}")
        
        return results


def main():
    print("=" * 60)
    print("BlurImageProcessor - 自定义管道示例")
    print("=" * 60)
    
    pipeline = CustomPipeline()
    
    print("\n创建测试图像...")
    test_image = create_test_image()
    test_path = Path(__file__).parent.parent / "data" / "test_image.png"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_image)
    print(f"测试图像保存到: {test_path}")
    
    print("\n添加模糊...")
    blurred = add_blur(test_image, blur_type='motion', kernel_size=15)
    blurred_path = Path(__file__).parent.parent / "data" / "blurred_image.png"
    cv2.imwrite(str(blurred_path), blurred)
    print(f"模糊图像保存到: {blurred_path}")
    
    print("\n" + "-" * 60)
    print("示例 1: 自定义处理步骤")
    print("-" * 60)
    
    custom_steps = [
        {'name': 'detect_blur', 'method': 'laplacian', 'params': {}},
        {'name': 'deblur', 'method': 'wiener', 'params': {'balance': 0.1}},
        {'name': 'enhance', 'method': 'sharpening', 'params': {'amount': 1.5}},
        {'name': 'evaluate', 'method': None, 'params': {}}
    ]
    
    results1 = pipeline.process_with_custom_steps(str(blurred_path), custom_steps)
    try:
        pipeline.visualizer.show_processing_pipeline(
            results1['images'],
            results1['titles'],
            results1['metrics']
        )
    except Exception as e:
        print(f"可视化错误: {e}")
        print("继续执行...")
    
    print("\n" + "-" * 60)
    print("示例 2: 比较不同去模糊方法")
    print("-" * 60)
    
    methods = {
        'wiener': {'balance': 0.1},
        'richardson_lucy': {'iterations': 30},
        'unsharp': {'sigma': 1.0, 'strength': 1.2}
    }
    
    results2 = pipeline.compare_methods(str(blurred_path), methods)
    try:
        pipeline.visualizer.show_comparison(
            results2['images'],
            results2['titles']
        )
    except Exception as e:
        print(f"可视化错误: {e}")
        print("继续执行...")
    
    print("\n" + "=" * 60)
    print("自定义管道示例完成!")
    print("=" * 60)


def create_test_image():
    image = np.zeros((512, 512), dtype=np.uint8)
    
    cv2.putText(image, "Custom Pipeline", (120, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    cv2.rectangle(image, (100, 250), (400, 350), 200, 2)
    cv2.circle(image, (250, 300), 30, 150, -1)
    
    for i in range(50, 450, 40):
        cv2.line(image, (i, 380), (i + 25, 380), 255, 2)
    
    return image


def add_blur(image, blur_type='motion', kernel_size=15):
    if blur_type == 'motion':
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[center, i] = 1
        kernel = kernel / kernel.sum()
    else:
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


if __name__ == "__main__":
    main()
