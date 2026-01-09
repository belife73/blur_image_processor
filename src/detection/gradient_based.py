import cv2
import numpy as np
from typing import Tuple, Optional
from utils.logger import Logger


class GradientDetector:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.threshold = self.config.get('threshold', 50.0)
        self.sobel_kernel = self.config.get('sobel_kernel', 3)
    
    def detect(self, image: np.ndarray,
               threshold: Optional[float] = None) -> Tuple[bool, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_variance = gradient_magnitude.var()
        
        if threshold is None:
            threshold = self.threshold
        
        is_blurry = gradient_variance < threshold
        
        self.logger.info(f"Gradient variance: {gradient_variance:.2f}, threshold: {threshold}")
        
        return is_blurry, float(gradient_variance)
    
    def get_gradient_map(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return gradient_magnitude
    
    def get_gradient_direction(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        return gradient_direction
    
    def get_edge_density(self, image: np.ndarray, 
                         edge_threshold: float = 50.0) -> float:
        gradient_map = self.get_gradient_map(image)
        edge_pixels = np.sum(gradient_map > edge_threshold)
        total_pixels = gradient_map.size
        
        edge_density = edge_pixels / total_pixels
        return float(edge_density)
    
    def detect_edges(self, image: np.ndarray, 
                     method: str = 'sobel',
                     threshold1: float = 50.0,
                     threshold2: float = 150.0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.uint8(np.clip(edges, 0, 255))
        
        elif method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2)
        
        elif method == 'prewitt':
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
            edges = np.uint8(np.clip(edges, 0, 255))
        
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        return edges
    
    def get_gradient_statistics(self, image: np.ndarray) -> dict:
        gradient_map = self.get_gradient_map(image)
        
        stats = {
            'mean': float(gradient_map.mean()),
            'std': float(gradient_map.std()),
            'min': float(gradient_map.min()),
            'max': float(gradient_map.max()),
            'median': float(np.median(gradient_map)),
            'variance': float(gradient_map.var())
        }
        
        return stats
