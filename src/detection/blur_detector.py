import cv2
import numpy as np
from typing import Tuple, Optional
from detection.laplacian import LaplacianDetector
from detection.fft_analysis import FFTDetector
from detection.gradient_based import GradientDetector
from utils.logger import Logger


class BlurDetector:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        
        self.laplacian_detector = LaplacianDetector(
            self.config.get('laplacian', {})
        )
        self.fft_detector = FFTDetector(
            self.config.get('fft', {})
        )
        self.gradient_detector = GradientDetector(
            self.config.get('gradient', {})
        )
    
    def detect_laplacian(self, image: np.ndarray, 
                         threshold: Optional[float] = None) -> Tuple[bool, float]:
        return self.laplacian_detector.detect(image, threshold)
    
    def detect_fft(self, image: np.ndarray,
                   threshold: Optional[float] = None) -> Tuple[bool, float]:
        return self.fft_detector.detect(image, threshold)
    
    def detect_gradient(self, image: np.ndarray,
                        threshold: Optional[float] = None) -> Tuple[bool, float]:
        return self.gradient_detector.detect(image, threshold)
    
    def detect_all(self, image: np.ndarray,
                   threshold: Optional[float] = None) -> dict:
        results = {
            'laplacian': self.detect_laplacian(image, threshold),
            'fft': self.detect_fft(image, threshold),
            'gradient': self.detect_gradient(image, threshold)
        }
        
        blurry_votes = sum(1 for is_blur, _ in results.values() if is_blur)
        results['consensus'] = blurry_votes >= 2
        
        self.logger.info(f"Detection results: {results}")
        return results
    
    def get_blur_level(self, image: np.ndarray) -> str:
        is_blurry, score = self.detect_laplacian(image)
        
        if score < 50:
            return "severely_blurry"
        elif score < 100:
            return "moderately_blurry"
        elif score < 200:
            return "slightly_blurry"
        else:
            return "sharp"
