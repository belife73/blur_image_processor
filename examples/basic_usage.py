import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.pipeline import BlurProcessor
from detection.blur_detector import BlurDetector
from utils.visualization import Visualizer


def main():
    print("=" * 60)
    print("BlurImageProcessor - 基础使用示例")
    print("=" * 60)
    
    processor = BlurProcessor()
    visualizer = Visualizer()
    
    print("\n1. 创建测试图像...")
    test_image = create_test_image()
    print(f"   测试图像创建完成，尺寸: {test_image.shape}")
    
    print("\n2. 添加模糊...")
    blurred_image = add_motion_blur(test_image, kernel_size=15, angle=45)
    print(f"   模糊图像创建完成")
    
    print("\n3. 加载模糊图像...")
    processor.load_from_array(blurred_image)
    
    print("\n4. 检测模糊...")
    is_blurry, score = processor.detect_blur(method='laplacian')
    print(f"   模糊检测结果: {'模糊' if is_blurry else '清晰'}")
    print(f"   模糊分数: {score:.2f}")
    
    print("\n5. 去模糊处理...")
    restored = processor.deblur(method='wiener', balance=0.1)
    print(f"   去模糊完成")
    
    print("\n6. 图像增强...")
    enhanced = processor.enhance(method='sharpening', amount=1.5)
    print(f"   图像增强完成")
    
    print("\n7. 质量评估...")
    metrics = processor.evaluate_quality()
    print("   质量指标:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    print("\n8. 可视化结果...")
    images = [test_image, blurred_image, enhanced]
    titles = ["原始图像", "模糊图像", "处理后图像"]
    visualizer.show_comparison(images, titles)
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)


def create_test_image(size=(512, 512)):
    image = np.zeros((size[0], size[1]), dtype=np.uint8)
    
    cv2.putText(image, "Blur Test", (150, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    cv2.rectangle(image, (100, 250), (400, 350), 200, 2)
    cv2.circle(image, (250, 300), 30, 150, -1)
    
    for i in range(50, 450, 50):
        cv2.line(image, (i, 380), (i + 30, 380), 255, 2)
    
    return image


def add_motion_blur(image, kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size))
    
    center = kernel_size // 2
    for i in range(kernel_size):
        offset = int(i * np.tan(np.deg2rad(angle)))
        if 0 <= center + offset < kernel_size:
            kernel[center + offset, i] = 1
    
    kernel = kernel / kernel.sum()
    
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


if __name__ == "__main__":
    main()
