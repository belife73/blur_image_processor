import cv2
import numpy as np
from typing import Optional
from utils.logger import Logger


class ContrastEnhancer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.alpha = self.config.get('alpha', 1.2)
        self.beta = self.config.get('beta', 10)
    
    def enhance(self, image,
                alpha: Optional[float] = None,
                beta: Optional[float] = None) -> np.ndarray:
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        self.logger.info(f"Contrast enhancement completed, alpha: {alpha}, beta: {beta}")
        return enhanced
    
    def histogram_equalization(self, image,
                                clip_limit: Optional[float] = None) -> np.ndarray:
        if len(image.shape) == 2:
            if clip_limit is not None:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                result = clahe.apply(image)
            else:
                result = cv2.equalizeHist(image)
        else:
            if clip_limit is not None:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                channels = [clahe.apply(channel) for channel in cv2.split(image)]
                result = cv2.merge(channels)
            else:
                channels = [cv2.equalizeHist(channel) for channel in cv2.split(image)]
                result = cv2.merge(channels)
        
        self.logger.info(f"Histogram equalization completed")
        return result
    
    def gamma_correction(self, image,
                         gamma: float = 1.0) -> np.ndarray:
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 
                                 for i in range(256)], dtype=np.uint8)
        result = cv2.LUT(image, lookup_table)
        
        self.logger.info(f"Gamma correction completed, gamma: {gamma}")
        return result
    
    def adaptive_contrast(self, image,
                           window_size: int = 8,
                           clip_limit: float = 2.0) -> np.ndarray:
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(window_size, window_size))
            result = clahe.apply(image)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(window_size, window_size))
            channels = [clahe.apply(channel) for channel in cv2.split(image)]
            result = cv2.merge(channels)
        
        self.logger.info(f"Adaptive contrast enhancement completed")
        return result
    
    def stretch_contrast(self, image,
                         in_range: Optional[tuple] = None,
                         out_range: tuple = (0, 255)) -> np.ndarray:
        if in_range is None:
            in_range = (image.min(), image.max())
        
        result = cv2.normalize(image, None, out_range[0], out_range[1],
                               cv2.NORM_MINMAX)
        
        self.logger.info(f"Contrast stretching completed")
        return result
    
    def brightness_contrast_adjust(self, image,
                                    brightness: float = 0,
                                    contrast: float = 0) -> np.ndarray:
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            
            image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
        
        if brightness != 0:
            image = cv2.add(image, brightness)
        
        result = np.clip(image, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Brightness and contrast adjustment completed")
        return result
    
    def local_contrast_normalization(self, image,
                                       kernel_size: int = 3) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        std = cv2.GaussianBlur(gray**2, (kernel_size, kernel_size), 0)
        std = np.sqrt(std - mean**2)
        
        normalized = (gray - mean) / (std + 1e-6)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        result = (normalized * 255).astype(np.uint8)
        
        self.logger.info(f"Local contrast normalization completed")
        return result
