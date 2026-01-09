import cv2
import numpy as np
from typing import Optional
from utils.logger import Logger


class NoiseReducer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.method = self.config.get('method', 'gaussian')
        self.kernel_size = self.config.get('kernel_size', 3)
    
    def enhance(self, image,
                method: Optional[str] = None,
                **kwargs) -> np.ndarray:
        if method is None:
            method = self.method
        
        if method == 'gaussian':
            return self.gaussian_blur(image, **kwargs)
        elif method == 'median':
            return self.median_blur(image, **kwargs)
        elif method == 'bilateral':
            return self.bilateral_filter(image, **kwargs)
        elif method == 'non_local_means':
            return self.non_local_means(image, **kwargs)
        elif method == 'morphological':
            return self.morphological_filter(image, **kwargs)
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
    
    def gaussian_blur(self, image,
                       kernel_size: Optional[int] = None,
                       sigma: float = 0) -> np.ndarray:
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        self.logger.info(f"Gaussian blur completed, kernel_size: {kernel_size}")
        return result
    
    def median_blur(self, image,
                     kernel_size: Optional[int] = None) -> np.ndarray:
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = cv2.medianBlur(image, kernel_size)
        
        self.logger.info(f"Median blur completed, kernel_size: {kernel_size}")
        return result
    
    def bilateral_filter(self, image,
                          d: int = 9,
                          sigma_color: float = 75,
                          sigma_space: float = 75) -> np.ndarray:
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        self.logger.info(f"Bilateral filter completed")
        return result
    
    def non_local_means(self, image,
                         h: float = 10,
                         template_window_size: int = 7,
                         search_window_size: int = 21) -> np.ndarray:
        if len(image.shape) == 2:
            result = cv2.fastNlMeansDenoising(image, None, h,
                                                template_window_size, search_window_size)
        else:
            result = cv2.fastNlMeansDenoisingColored(image, None, h, h,
                                                      template_window_size, search_window_size)
        
        self.logger.info(f"Non-local means denoising completed")
        return result
    
    def morphological_filter(self, image,
                              operation: str = 'opening',
                              kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if operation == 'opening':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'erosion':
            result = cv2.erode(image, kernel)
        elif operation == 'dilation':
            result = cv2.dilate(image, kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
        
        self.logger.info(f"Morphological filter completed, operation: {operation}")
        return result
    
    def wiener_filter(self, image,
                       noise_variance: float = 0.01,
                       signal_variance: float = 1.0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols), np.float32)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        
        h = mask
        h_conj = np.conj(h)
        
        H_mag_sq = np.abs(h) ** 2
        wiener_filter = h_conj / (H_mag_sq + noise_variance / signal_variance)
        
        g_shift = fshift * wiener_filter
        g = np.fft.ifftshift(g_shift)
        result = np.fft.ifft2(g)
        result = np.abs(result)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Wiener filter denoising completed")
        return result
    
    def adaptive_bilateral_filter(self, image,
                                    d: int = 9,
                                    sigma_color: float = 75,
                                    sigma_space: float = 75,
                                    max_sigma: float = 150) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        local_std = cv2.GaussianBlur(gray**2, (d, d), sigma_space) - \
                   cv2.GaussianBlur(gray, (d, d), sigma_space)**2
        local_std = np.sqrt(np.maximum(local_std, 0))
        
        adaptive_sigma = np.clip(local_std, sigma_color, max_sigma)
        
        result = np.zeros_like(gray)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                sigma = adaptive_sigma[i, j]
                result[i, j] = cv2.bilateralFilter(gray, d, sigma, sigma_space)[i, j]
        
        self.logger.info(f"Adaptive bilateral filter completed")
        return result
