import numpy as np
from skimage import restoration, img_as_float, img_as_ubyte
from typing import Optional
from utils.logger import Logger


class RichardsonLucy:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.iterations = self.config.get('iterations', 30)
        self.psf_size = self.config.get('psf_size', 5)
    
    def deblur(self, image,
                iterations: Optional[int] = None,
                psf: Optional[np.ndarray] = None,
                psf_size: Optional[int] = None,
                damping: float = 0.0) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        img_float = img_as_float(gray)
        
        if iterations is None:
            iterations = self.iterations
        
        if psf is None:
            if psf_size is None:
                psf_size = self.psf_size
            psf = self._create_default_psf(psf_size)
        
        deconvolved = restoration.richardson_lucy(
            img_float, psf, num_iter=iterations, clip=True
        )
        
        if damping > 0:
            deconvolved = self._apply_damping(deconvolved, damping)
        
        deconvolved = np.clip(deconvolved, 0, 1)
        result = img_as_ubyte(deconvolved)
        
        self.logger.info(f"Richardson-Lucy deblurring completed, iterations: {iterations}")
        return result
    
    def _create_default_psf(self, size: int) -> np.ndarray:
        psf = np.ones((size, size)) / (size * size)
        return psf
    
    def _apply_damping(self, image: np.ndarray, damping: float) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(image, sigma=damping)
        return smoothed
    
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
    
    def adaptive_iterations(self, image: np.ndarray,
                            max_iterations: int = 50,
                            min_iterations: int = 10) -> int:
        from detection.laplacian import LaplacianDetector
        
        detector = LaplacianDetector()
        _, blur_score = detector.detect(image)
        
        if blur_score < 50:
            return max_iterations
        elif blur_score < 100:
            return (max_iterations + min_iterations) // 2
        else:
            return min_iterations
