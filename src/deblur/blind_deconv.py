import numpy as np
from skimage import restoration, img_as_float, img_as_ubyte
from typing import Optional
from utils.logger import Logger


class BlindDeconvolution:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.iterations = self.config.get('iterations', 50)
        self.psf_size = self.config.get('psf_size', 7)
    
    def deblur(self, image,
                iterations: Optional[int] = None,
                psf_size: Optional[int] = None,
                reg: float = 0.1) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        img_float = img_as_float(gray)
        
        if iterations is None:
            iterations = self.iterations
        
        if psf_size is None:
            psf_size = self.psf_size
        
        psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)
        
        deconvolved, estimated_psf = restoration.unsupervised_wiener(
            img_float, psf, reg=reg, max_num_iter=iterations
        )
        
        deconvolved = np.clip(deconvolved, 0, 1)
        result = img_as_ubyte(deconvolved)
        
        self.logger.info(f"Blind deconvolution completed, iterations: {iterations}")
        self.logger.info(f"Estimated PSF shape: {estimated_psf.shape}")
        
        return result
    
    def estimate_psf(self, image,
                     psf_size: Optional[int] = None,
                     iterations: Optional[int] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        img_float = img_as_float(gray)
        
        if psf_size is None:
            psf_size = self.psf_size
        
        if iterations is None:
            iterations = self.iterations
        
        initial_psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)
        
        _, estimated_psf = restoration.unsupervised_wiener(
            img_float, initial_psf, reg=0.1, max_num_iter=iterations
        )
        
        return estimated_psf
    
    def analyze_psf(self, psf: np.ndarray) -> dict:
        center = np.unravel_index(np.argmax(psf), psf.shape)
        
        y_coords, x_coords = np.indices(psf.shape)
        y_center, x_center = center
        
        distances = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
        weighted_dist = distances * psf
        avg_radius = np.sum(weighted_dist) / np.sum(psf)
        
        psf_analysis = {
            'shape': psf.shape,
            'center': center,
            'max_value': float(psf.max()),
            'mean_value': float(psf.mean()),
            'std_value': float(psf.std()),
            'estimated_radius': float(avg_radius)
        }
        
        return psf_analysis
    
    def detect_blur_type(self, psf: np.ndarray) -> str:
        center = np.unravel_index(np.argmax(psf), psf.shape)
        
        vertical_profile = psf[:, center[1]]
        horizontal_profile = psf[center[0], :]
        
        vertical_std = vertical_profile.std()
        horizontal_std = horizontal_profile.std()
        
        ratio = vertical_std / (horizontal_std + 1e-6)
        
        if ratio > 2.0 or ratio < 0.5:
            return 'motion_blur'
        elif ratio > 0.8 and ratio < 1.2:
            return 'gaussian_blur'
        else:
            return 'unknown'
