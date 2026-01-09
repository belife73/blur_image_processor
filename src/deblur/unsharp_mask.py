import cv2
import numpy as np
from typing import Optional
from utils.logger import Logger


class UnsharpMask:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.sigma = self.config.get('sigma', 1.0)
        self.strength = self.config.get('strength', 1.5)
    
    def deblur(self, image,
                sigma: Optional[float] = None,
                strength: Optional[float] = None,
                threshold: float = 0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        if sigma is None:
            sigma = self.sigma
        
        if strength is None:
            strength = self.strength
        
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        
        if threshold > 0:
            mask = np.abs(gray - blurred) > threshold
            sharpened = np.where(mask, sharpened, gray)
        
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Unsharp mask completed, sigma: {sigma}, strength: {strength}")
        return result
    
    def adaptive_unsharp_mask(self, image,
                              base_sigma: float = 1.0,
                              base_strength: float = 1.5,
                              edge_threshold: float = 30.0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        edge_mask = edge_magnitude > edge_threshold
        edge_ratio = np.sum(edge_mask) / edge_mask.size
        
        adaptive_sigma = base_sigma * (1.0 - edge_ratio * 0.5)
        adaptive_strength = base_strength * (1.0 + edge_ratio * 0.5)
        
        result = self.deblur(gray, sigma=adaptive_sigma, strength=adaptive_strength)
        
        return result
    
    def multi_scale_unsharp_mask(self, image,
                                  sigmas: list = None,
                                  strengths: list = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        if sigmas is None:
            sigmas = [0.5, 1.0, 2.0]
        
        if strengths is None:
            strengths = [0.5, 1.0, 0.5]
        
        if len(sigmas) != len(strengths):
            raise ValueError("sigmas and strengths must have the same length")
        
        result = gray.astype(np.float32)
        
        for sigma, strength in zip(sigmas, strengths):
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
            result = (result + sharpened) / 2
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Multi-scale unsharp mask completed")
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
