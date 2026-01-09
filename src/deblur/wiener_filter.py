import numpy as np
from skimage import restoration, img_as_float, img_as_ubyte
from typing import Optional, Tuple
from utils.logger import Logger


class WienerFilter:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.balance = self.config.get('balance', 0.1)
        self.psf_size = self.config.get('psf_size', 5)
    
    def deblur(self, image, 
                balance: Optional[float] = None,
                psf: Optional[np.ndarray] = None,
                psf_size: Optional[int] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        img_float = img_as_float(gray)
        
        if balance is None:
            balance = self.balance
        
        if psf is None:
            if psf_size is None:
                psf_size = self.psf_size
            psf = self._create_default_psf(psf_size)
        
        deconvolved = restoration.wiener(img_float, psf, balance=balance)
        
        deconvolved = np.clip(deconvolved, 0, 1)
        result = img_as_ubyte(deconvolved)
        
        self.logger.info(f"Wiener deblurring completed, balance: {balance}")
        return result
    
    def _create_default_psf(self, size: int) -> np.ndarray:
        psf = np.ones((size, size)) / (size * size)
        return psf
    
    def create_motion_psf(self, length: int, angle: float) -> np.ndarray:
        psf = np.zeros((length, length))
        center = length // 2
        
        for i in range(length):
            offset = int(i * np.tan(np.deg2rad(angle)))
            if 0 <= center + offset < length:
                psf[center + offset, i] = 1
        
        psf = psf / psf.sum()
        return psf
    
    def create_gaussian_psf(self, size: int, sigma: float = 1.0) -> np.ndarray:
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf = psf / psf.sum()
        return psf
    
    def estimate_noise(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        h, w = gray.shape
        m = int(h * 0.1)
        n = int(w * 0.1)
        
        noise_region = gray[m:h-m, n:w-n]
        noise_std = noise_region.std()
        
        return float(noise_std)
    
    def auto_balance(self, image: np.ndarray) -> float:
        noise_std = self.estimate_noise(image)
        balance = max(0.01, min(1.0, noise_std / 50.0))
        return balance
