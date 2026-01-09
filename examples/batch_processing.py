import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.pipeline import BlurProcessor
from utils.logger import Logger


def main():
    print("=" * 60)
    print("BlurImageProcessor - 批量处理示例")
    print("=" * 60)
    
    logger = Logger.get_logger()
    processor = BlurProcessor()
    
    input_dir = Path(__file__).parent.parent / "data" / "sample_images"
    output_dir = Path(__file__).parent.parent / "data" / "processed_images"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    if not input_dir.exists():
        print(f"\n警告: 输入目录不存在，创建测试图像...")
        input_dir.mkdir(parents=True, exist_ok=True)
        create_sample_images(input_dir)
    
    print("\n扫描图像文件...")
    image_files = list(input_dir.glob("*.jpg")) + \
                  list(input_dir.glob("*.png")) + \
                  list(input_dir.glob("*.bmp"))
    
    if not image_files:
        print("未找到图像文件!")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理 [{i}/{len(image_files)}]: {image_path.name}")
        
        try:
            processor.load_image(image_path)
            
            is_blurry, score = processor.detect_blur(method='laplacian')
            
            if is_blurry:
                print(f"  图像模糊 (分数: {score:.2f})，开始处理...")
                
                processor.deblur(method='wiener', balance=0.1)
                processor.enhance(method='sharpening', amount=1.2)
                
                output_path = output_dir / f"processed_{image_path.name}"
                processor.save_result(output_path)
                
                metrics = processor.evaluate_quality()
                
                result = {
                    'filename': image_path.name,
                    'blur_score': score,
                    'output': str(output_path),
                    'metrics': metrics
                }
                results.append(result)
                
                print(f"  处理完成，保存到: {output_path}")
                print(f"  PSNR: {metrics.get('PSNR', 0):.2f}, SSIM: {metrics.get('SSIM', 0):.4f}")
            else:
                print(f"  图像清晰 (分数: {score:.2f})，跳过处理")
                result = {
                    'filename': image_path.name,
                    'blur_score': score,
                    'output': None,
                    'metrics': {}
                }
                results.append(result)
            
            processor.reset()
            
        except Exception as e:
            print(f"  错误: {e}")
            logger.error(f"处理 {image_path} 时出错: {e}")
    
    print("\n" + "=" * 60)
    print("批量处理完成!")
    print("=" * 60)
    
    print("\n处理摘要:")
    print(f"总文件数: {len(image_files)}")
    print(f"处理文件数: {len([r for r in results if r['output']])}")
    print(f"跳过文件数: {len([r for r in results if not r['output']])}")
    
    print("\n详细结果:")
    for result in results:
        status = "已处理" if result['output'] else "已跳过"
        print(f"  {result['filename']}: {status} (模糊分数: {result['blur_score']:.2f})")


def create_sample_images(directory):
    import cv2
    import numpy as np
    
    print("创建示例图像...")
    
    for i in range(3):
        image = np.zeros((512, 512), dtype=np.uint8)
        
        cv2.putText(image, f"Sample {i+1}", (150, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.rectangle(image, (100, 250), (400, 350), 200, 2)
        cv2.circle(image, (250, 300), 30, 150, -1)
        
        if i == 1:
            kernel = np.ones((10, 10)) / 100
            image = cv2.filter2D(image, -1, kernel)
        elif i == 2:
            kernel = np.ones((15, 15)) / 225
            image = cv2.filter2D(image, -1, kernel)
        
        output_path = directory / f"sample_{i+1}.png"
        cv2.imwrite(str(output_path), image)
        print(f"  创建: {output_path.name}")


if __name__ == "__main__":
    main()
