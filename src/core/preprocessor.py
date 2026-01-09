import cv2
import numpy as np
from typing import Optional, Tuple, Union
from utils.logger import Logger


class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger()
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def normalize(self, image: np.ndarray, 
                  target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        img_min, img_max = image.min(), image.max()
        if img_max - img_min == 0:
            return np.full_like(image, target_range[0])
        
        normalized = (image - img_min) / (img_max - img_min)
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        return normalized
    
    def denormalize(self, image: np.ndarray,
                    original_range: Tuple[float, float] = (0.0, 255.0)) -> np.ndarray:
        img_min, img_max = image.min(), image.max()
        if img_max - img_min == 0:
            return np.full_like(image, original_range[0])
        
        denormalized = (image - img_min) / (img_max - img_min)
        denormalized = denormalized * (original_range[1] - original_range[0]) + original_range[0]
        return denormalized.astype(np.uint8)
    
    def resize(self, image: np.ndarray,
               size: Optional[Tuple[int, int]] = None,
               scale: Optional[float] = None,
               keep_aspect: bool = True) -> np.ndarray:
        h, w = image.shape[:2]
        
        if scale is not None:
            new_w, new_h = int(w * scale), int(h * scale)
        elif size is not None:
            new_w, new_h = size
            if keep_aspect:
                aspect = w / h
                if new_w / new_h > aspect:
                    new_w = int(new_h * aspect)
                else:
                    new_h = int(new_w / aspect)
        else:
            return image
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def crop_center(self, image: np.ndarray, 
                    crop_size: Tuple[int, int]) -> np.ndarray:
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        if crop_h > h or crop_w > w:
            raise ValueError("Crop size is larger than image size")
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    def pad_image(self, image: np.ndarray,
                  padding: Union[int, Tuple[int, int, int, int]],
                  mode: str = 'constant',
                  value: int = 0) -> np.ndarray:
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        
        top, bottom, left, right = padding
        
        if mode == 'constant':
            return cv2.copyMakeBorder(image, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=value)
        elif mode == 'reflect':
            return cv2.copyMakeBorder(image, top, bottom, left, right,
                                     cv2.BORDER_REFLECT)
        elif mode == 'replicate':
            return cv2.copyMakeBorder(image, top, bottom, left, right,
                                     cv2.BORDER_REPLICATE)
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")
    
    def adjust_brightness(self, image: np.ndarray, 
                          factor: float = 1.0) -> np.ndarray:
        if factor == 1.0:
            return image
        
        adjusted = image.astype(np.float32) * factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def adjust_contrast(self, image: np.ndarray,
                        factor: float = 1.0) -> np.ndarray:
        if factor == 1.0:
            return image
        
        mean = image.mean()
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def gamma_correction(self, image: np.ndarray,
                        gamma: float = 1.0) -> np.ndarray:
        if gamma == 1.0:
            return image
        
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 
                                for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, lookup_table)
    
    def histogram_equalization(self, image: np.ndarray,
                              clip_limit: Optional[float] = None) -> np.ndarray:
        if len(image.shape) == 2:
            if clip_limit is not None:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                return clahe.apply(image)
            return cv2.equalizeHist(image)
        else:
            if clip_limit is not None:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                channels = [clahe.apply(channel) for channel in cv2.split(image)]
                return cv2.merge(channels)
            else:
                channels = [cv2.equalizeHist(channel) for channel in cv2.split(image)]
                return cv2.merge(channels)
    
    def bilateral_filter(self, image: np.ndarray,
                        d: int = 9,
                        sigma_color: float = 75,
                        sigma_space: float = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def gaussian_blur(self, image: np.ndarray,
                     kernel_size: int = 5,
                     sigma: float = 0) -> np.ndarray:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def median_filter(self, image: np.ndarray,
                      kernel_size: int = 5) -> np.ndarray:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)
    
    def preprocess_pipeline(self, image: np.ndarray,
                            steps: list) -> np.ndarray:
        result = image.copy()
        
        for step in steps:
            method = step.get('method')
            params = step.get('params', {})
            
            if hasattr(self, method):
                result = getattr(self, method)(result, **params)
            else:
                self.logger.warning(f"Unknown preprocessing method: {method}")
        
        return result
