import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from core.image_loader import ImageLoader
from core.preprocessor import Preprocessor
from detection.blur_detector import BlurDetector
from deblur.deblur_engine import DeblurEngine
from enhancement.enhancer import ImageEnhancer
from analysis.quality_metrics import QualityMetrics
from utils.logger import Logger
from utils.visualization import Visualizer


class BlurProcessor:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config = self._load_config(config_path)
        self.logger = Logger.get_logger()
        
        self.loader = ImageLoader()
        self.preprocessor = Preprocessor(self.logger)
        self.detector = BlurDetector(self.config.get('detection', {}))
        self.deblur_engine = DeblurEngine(self.config.get('deblur', {}))
        self.enhancer = ImageEnhancer(self.config.get('enhancement', {}))
        self.metrics = QualityMetrics(self.config.get('analysis', {}))
        self.visualizer = Visualizer()
        
        self.image = None
        self.gray_image = None
        self.processed_image = None
        self.original_path = None
        
        self.logger.info("BlurProcessor initialized")
    
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_image(self, path: Union[str, Path],
                   color_mode: str = 'BGR') -> np.ndarray:
        self.image = self.loader.load(path, color_mode=color_mode)
        self.gray_image = self.preprocessor.to_grayscale(self.image)
        self.original_path = Path(path)
        self.logger.info(f"Image loaded: {path}")
        return self.image
    
    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        self.image = self.loader.load_from_array(array)
        self.gray_image = self.preprocessor.to_grayscale(self.image)
        self.original_path = None
        self.logger.info("Image loaded from array")
        return self.image
    
    def detect_blur(self, method: str = 'laplacian', **kwargs) -> tuple:
        if self.gray_image is None:
            raise ValueError("No image loaded")
        
        if method == 'laplacian':
            is_blurry, score = self.detector.detect_laplacian(self.gray_image, **kwargs)
        elif method == 'fft':
            is_blurry, score = self.detector.detect_fft(self.gray_image, **kwargs)
        elif method == 'gradient':
            is_blurry, score = self.detector.detect_gradient(self.gray_image, **kwargs)
        elif method == 'all':
            results = self.detector.detect_all(self.gray_image, **kwargs)
            return results
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        self.logger.info(f"Blur detection ({method}): blurry={is_blurry}, score={score:.2f}")
        return is_blurry, score
    
    def deblur(self, method: str = 'wiener', **kwargs) -> np.ndarray:
        if self.gray_image is None:
            raise ValueError("No image loaded")
        
        self.processed_image = self.deblur_engine.deblur(
            self.gray_image, method=method, **kwargs
        )
        
        self.logger.info(f"Deblurring completed using {method}")
        return self.processed_image
    
    def enhance(self, method: str = 'sharpening', **kwargs) -> np.ndarray:
        if self.processed_image is None:
            image_to_enhance = self.gray_image
        else:
            image_to_enhance = self.processed_image
        
        enhanced = self.enhancer.enhance(image_to_enhance, method=method, **kwargs)
        self.processed_image = enhanced
        
        self.logger.info(f"Enhancement completed using {method}")
        return enhanced
    
    def evaluate_quality(self, reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        if reference is None:
            reference = self.gray_image
        
        metrics = self.metrics.calculate_all(self.processed_image, reference)
        self.logger.info(f"Quality metrics: {metrics}")
        return metrics
    
    def process(self, 
                detect_method: str = 'laplacian',
                deblur_method: str = 'wiener',
                enhance_method: str = 'sharpening',
                auto_deblur: bool = True,
                **kwargs) -> Dict[str, Any]:
        results = {
            'original': self.image,
            'gray': self.gray_image,
            'is_blurry': False,
            'blur_score': 0.0,
            'processed': None,
            'metrics': {}
        }
        
        is_blurry, score = self.detect_blur(method=detect_method, **kwargs)
        results['is_blurry'] = is_blurry
        results['blur_score'] = score
        
        if is_blurry or not auto_deblur:
            self.deblur(method=deblur_method, **kwargs)
            self.enhance(method=enhance_method, **kwargs)
            results['processed'] = self.processed_image
            results['metrics'] = self.evaluate_quality()
        
        return results
    
    def save_result(self, path: Union[str, Path], 
                    image: Optional[np.ndarray] = None,
                    quality: int = 95) -> None:
        if image is None:
            image = self.processed_image
        
        if image is None:
            raise ValueError("No image to save")
        
        self.loader.save(image, path, quality=quality)
        self.logger.info(f"Result saved to: {path}")
    
    def visualize_results(self, show: bool = True) -> None:
        if self.image is None:
            raise ValueError("No image loaded")
        
        images = [self.gray_image]
        titles = ["Original"]
        
        if self.processed_image is not None:
            images.append(self.processed_image)
            titles.append("Processed")
        
        self.visualizer.show_comparison(images, titles, show=show)
    
    def get_image_info(self) -> Dict[str, Any]:
        if self.image is None:
            raise ValueError("No image loaded")
        return self.loader.get_image_info(self.image)
    
    def reset(self) -> None:
        self.image = None
        self.gray_image = None
        self.processed_image = None
        self.original_path = None
        self.logger.info("Processor reset")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Blur Image Processor")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path", 
                       default="output.jpg")
    parser.add_argument("-c", "--config", help="Config file path",
                       default=None)
    parser.add_argument("-d", "--detect", help="Blur detection method",
                       choices=['laplacian', 'fft', 'gradient', 'all'],
                       default='laplacian')
    parser.add_argument("-b", "--deblur", help="Deblurring method",
                       choices=['wiener', 'richardson_lucy', 'blind', 'unsharp'],
                       default='wiener')
    parser.add_argument("-e", "--enhance", help="Enhancement method",
                       choices=['sharpening', 'contrast', 'denoise'],
                       default='sharpening')
    parser.add_argument("--no-auto", help="Process even if image is not blurry",
                       action="store_true")
    parser.add_argument("--show", help="Show visualization",
                       action="store_true")
    
    args = parser.parse_args()
    
    processor = BlurProcessor(config_path=args.config)
    processor.load_image(args.input)
    
    results = processor.process(
        detect_method=args.detect,
        deblur_method=args.deblur,
        enhance_method=args.enhance,
        auto_deblur=not args.no_auto
    )
    
    print(f"Blur Score: {results['blur_score']:.2f}")
    print(f"Is Blurry: {results['is_blurry']}")
    
    if results['processed'] is not None:
        processor.save_result(args.output)
        print(f"Output saved to: {args.output}")
        
        if results['metrics']:
            print("Quality Metrics:")
            for metric, value in results['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        if args.show:
            processor.visualize_results()
    else:
        print("Image is sharp enough, no processing needed.")


if __name__ == "__main__":
    main()
