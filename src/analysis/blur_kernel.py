import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from utils.logger import Logger


class BlurKernelEstimator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.estimation_method = self.config.get('blur_kernel', {}).get('estimation_method', 'auto')
        self.max_kernel_size = self.config.get('blur_kernel', {}).get('max_kernel_size', 15)
    
    def estimate_psf(self, image: np.ndarray,
                       method: Optional[str] = None,
                       max_size: Optional[int] = None) -> np.ndarray:
        if method is None:
            method = self.estimation_method
        
        if max_size is None:
            max_size = self.max_kernel_size
        
        if method == 'auto':
            return self._auto_estimate(image, max_size)
        elif method == 'gradient':
            return self._gradient_based_estimate(image, max_size)
        elif method == 'spectrum':
            return self._spectrum_based_estimate(image, max_size)
        elif method == 'edge':
            return self._edge_based_estimate(image, max_size)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    
    def _auto_estimate(self, image: np.ndarray, max_size: int) -> np.ndarray:
        from ..detection.laplacian import LaplacianDetector
        
        detector = LaplacianDetector()
        _, blur_score = detector.detect(image)
        
        if blur_score < 50:
            kernel_size = min(max_size, 15)
        elif blur_score < 100:
            kernel_size = min(max_size, 11)
        elif blur_score < 200:
            kernel_size = min(max_size, 7)
        else:
            kernel_size = 5
        
        psf = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        self.logger.info(f"Auto-estimated PSF size: {kernel_size}x{kernel_size}")
        return psf
    
    def _gradient_based_estimate(self, image: np.ndarray, max_size: int) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_x = np.mean(np.abs(sobel_x))
        gradient_y = np.mean(np.abs(sobel_y))
        
        ratio = gradient_x / (gradient_y + 1e-6)
        
        if ratio > 1.5:
            kernel_size = min(max_size, int(15 * ratio))
            psf = np.zeros((3, kernel_size))
            psf[1, :] = 1.0 / kernel_size
        elif ratio < 0.67:
            kernel_size = min(max_size, int(15 / ratio))
            psf = np.zeros((kernel_size, 3))
            psf[:, 1] = 1.0 / kernel_size
        else:
            kernel_size = min(max_size, 7)
            psf = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        self.logger.info(f"Gradient-based PSF estimated, size: {psf.shape}")
        return psf
    
    def _spectrum_based_estimate(self, image: np.ndarray, max_size: int) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        horizontal_profile = magnitude[center_h, :]
        vertical_profile = magnitude[:, center_w]
        
        h_width = self._find_width(horizontal_profile, center_w)
        v_width = self._find_width(vertical_profile, center_h)
        
        kernel_h = min(max_size, max(3, int(v_width / 10)))
        kernel_w = min(max_size, max(3, int(h_width / 10)))
        
        psf = np.ones((kernel_h, kernel_w)) / (kernel_h * kernel_w)
        
        self.logger.info(f"Spectrum-based PSF estimated, size: {psf.shape}")
        return psf
    
    def _find_width(self, profile: np.ndarray, center: int) -> int:
        threshold = profile.max() * 0.5
        
        left = center
        while left > 0 and profile[left] > threshold:
            left -= 1
        
        right = center
        while right < len(profile) and profile[right] > threshold:
            right += 1
        
        return right - left
    
    def _edge_based_estimate(self, image: np.ndarray, max_size: int) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.ones((5, 5)) / 25
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:
            return np.ones((5, 5)) / 25
        
        ellipse = cv2.fitEllipse(largest_contour)
        
        major_axis = max(ellipse[1])
        kernel_size = min(max_size, int(major_axis / 5))
        kernel_size = max(3, kernel_size if kernel_size % 2 else kernel_size + 1)
        
        psf = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        self.logger.info(f"Edge-based PSF estimated, size: {psf.shape}")
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
    
    def create_defocus_psf(self, radius: float) -> np.ndarray:
        size = int(radius * 2 + 1)
        if size % 2 == 0:
            size += 1
        
        y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
        mask = x**2 + y**2 <= radius**2
        
        psf = mask.astype(np.float32)
        psf = psf / psf.sum()
        
        return psf
    
    def analyze_psf(self, psf: np.ndarray) -> Dict[str, any]:
        center = np.unravel_index(np.argmax(psf), psf.shape)
        
        y_coords, x_coords = np.indices(psf.shape)
        y_center, x_center = center
        
        distances = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
        weighted_dist = distances * psf
        avg_radius = np.sum(weighted_dist) / np.sum(psf)
        
        vertical_profile = psf[:, center[1]]
        horizontal_profile = psf[center[0], :]
        
        v_std = vertical_profile.std()
        h_std = horizontal_profile.std()
        
        analysis = {
            'shape': psf.shape,
            'center': center,
            'max_value': float(psf.max()),
            'mean_value': float(psf.mean()),
            'std_value': float(psf.std()),
            'estimated_radius': float(avg_radius),
            'aspect_ratio': float(v_std / (h_std + 1e-6)),
            'blur_type': self._classify_blur_type(v_std, h_std)
        }
        
        return analysis
    
    def _classify_blur_type(self, v_std: float, h_std: float) -> str:
        ratio = v_std / (h_std + 1e-6)
        
        if ratio > 2.0 or ratio < 0.5:
            return 'motion_blur'
        elif ratio > 0.8 and ratio < 1.2:
            return 'gaussian_blur'
        else:
            return 'unknown'
