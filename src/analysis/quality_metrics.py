import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional, Dict
from utils.logger import Logger


class QualityMetrics:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.use_psnr = self.config.get('quality_metrics', {}).get('use_psnr', True)
        self.use_ssim = self.config.get('quality_metrics', {}).get('use_ssim', True)
        self.use_mse = self.config.get('quality_metrics', {}).get('use_mse', True)
    
    def calculate_psnr(self, image1: np.ndarray, 
                        image2: np.ndarray) -> float:
        mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return float(psnr)
    
    def calculate_ssim(self, image1: np.ndarray,
                         image2: np.ndarray) -> float:
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        ssim_value = ssim(image1, image2, data_range=255)
        
        return float(ssim_value)
    
    def calculate_mse(self, image1: np.ndarray,
                       image2: np.ndarray) -> float:
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
        
        return float(mse)
    
    def calculate_mae(self, image1: np.ndarray,
                       image2: np.ndarray) -> float:
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        mae = np.mean(np.abs(image1.astype(np.float64) - image2.astype(np.float64)))
        
        return float(mae)
    
    def calculate_all(self, image1: np.ndarray,
                       image2: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics = {}
        
        if self.use_mse:
            if image2 is not None:
                metrics['MSE'] = self.calculate_mse(image1, image2)
            else:
                metrics['MSE'] = self.calculate_mse(image1, image1)
        
        if self.use_psnr:
            if image2 is not None:
                metrics['PSNR'] = self.calculate_psnr(image1, image2)
            else:
                metrics['PSNR'] = self.calculate_psnr(image1, image1)
        
        if self.use_ssim:
            if image2 is not None:
                metrics['SSIM'] = self.calculate_ssim(image1, image2)
            else:
                metrics['SSIM'] = self.calculate_ssim(image1, image1)
        
        metrics['MAE'] = self.calculate_mae(image1, image2 if image2 is not None else image1)
        
        return metrics
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contrast = gray.std()
        
        return float(contrast)
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        brightness = gray.mean()
        
        return float(brightness)
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    def calculate_no_reference_metrics(self, image: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['sharpness'] = self.calculate_sharpness(image)
        metrics['contrast'] = self.calculate_contrast(image)
        metrics['brightness'] = self.calculate_brightness(image)
        metrics['entropy'] = self.calculate_entropy(image)
        
        return metrics
    
    def compare_images(self, image1: np.ndarray,
                        image2: np.ndarray,
                        metrics_list: Optional[list] = None) -> Dict[str, float]:
        if metrics_list is None:
            metrics_list = ['PSNR', 'SSIM', 'MSE', 'MAE']
        
        results = {}
        
        for metric in metrics_list:
            if metric == 'PSNR':
                results[metric] = self.calculate_psnr(image1, image2)
            elif metric == 'SSIM':
                results[metric] = self.calculate_ssim(image1, image2)
            elif metric == 'MSE':
                results[metric] = self.calculate_mse(image1, image2)
            elif metric == 'MAE':
                results[metric] = self.calculate_mae(image1, image2)
        
        return results
