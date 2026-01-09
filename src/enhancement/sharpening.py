import cv2
import numpy as np
from typing import Optional
from utils.logger import Logger


class Sharpening:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.kernel_size = self.config.get('kernel_size', 3)
        self.amount = self.config.get('amount', 1.5)
    
    def enhance(self, image,
                kernel_size: Optional[int] = None,
                amount: Optional[float] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        if amount is None:
            amount = self.amount
        
        kernel = self._create_sharpening_kernel(kernel_size)
        sharpened = cv2.filter2D(gray, -1, kernel * amount)
        
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Sharpening completed, kernel_size: {kernel_size}, amount: {amount}")
        return result
    
    def _create_sharpening_kernel(self, size: int) -> np.ndarray:
        if size == 3:
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
        elif size == 5:
            kernel = np.array([[0, 0, -1, 0, 0],
                              [0, -1, -2, -1, 0],
                              [-1, -2, 16, -2, -1],
                              [0, -1, -2, -1, 0],
                              [0, 0, -1, 0, 0]])
        else:
            center = size // 2
            kernel = -np.ones((size, size))
            kernel[center, center] = size * size - 1
        
        return kernel
    
    def unsharp_mask(self, image,
                      sigma: float = 1.0,
                      amount: float = 1.5,
                      threshold: float = 0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
        
        if threshold > 0:
            mask = np.abs(gray - blurred) > threshold
            sharpened = np.where(mask, sharpened, gray)
        
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Unsharp mask completed, sigma: {sigma}, amount: {amount}")
        return result
    
    def laplacian_sharpening(self, image,
                              strength: float = 0.5) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpened = gray - strength * laplacian
        
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Laplacian sharpening completed, strength: {strength}")
        return result
    
    def high_boost_filter(self, image,
                           boost_factor: float = 2.0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = gray.astype(np.float32) - blurred.astype(np.float32)
        
        sharpened = gray.astype(np.float32) + boost_factor * mask
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        self.logger.info(f"High boost filter completed, boost factor: {boost_factor}")
        return result
    
    def adaptive_sharpening(self, image,
                             base_amount: float = 1.0,
                             edge_threshold: float = 30.0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        normalized_edges = edge_magnitude / (edge_magnitude.max() + 1e-6)
        adaptive_amount = base_amount * (1.0 + normalized_edges)
        
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharpened = gray.astype(np.float32) + adaptive_amount * (gray.astype(np.float32) - blurred.astype(np.float32))
        
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Adaptive sharpening completed")
        return result
