import cv2
import numpy as np
from typing import Tuple, Optional
from utils.logger import Logger


class LaplacianDetector:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.threshold = self.config.get('threshold', 100.0)
        self.kernel_size = self.config.get('kernel_size', 3)
    
    def detect(self, image: np.ndarray, 
               threshold: Optional[float] = None) -> Tuple[bool, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.kernel_size)
        variance = laplacian.var()
        
        if threshold is None:
            threshold = self.threshold
        
        is_blurry = variance < threshold
        
        self.logger.info(f"Laplacian variance: {variance:.2f}, threshold: {threshold}")
        
        return is_blurry, float(variance)
    
    def get_laplacian_map(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.kernel_size)
        return np.abs(laplacian)
    
    def get_edge_strength(self, image: np.ndarray) -> float:
        laplacian_map = self.get_laplacian_map(image)
        return float(laplacian_map.mean())
    
    def detect_local_blur(self, image: np.ndarray, 
                          window_size: int = 64) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        blur_map = np.zeros((h // window_size, w // window_size))
        
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = gray[i:i+window_size, j:j+window_size]
                if window.shape[0] == window_size and window.shape[1] == window_size:
                    laplacian = cv2.Laplacian(window, cv2.CV_64F)
                    blur_map[i//window_size, j//window_size] = laplacian.var()
        
        return blur_map
