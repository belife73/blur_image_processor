import cv2
import numpy as np
from typing import Tuple, Optional
from utils.logger import Logger


class FFTDetector:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        self.high_freq_ratio = self.config.get('high_freq_ratio', 0.3)
        self.threshold = self.config.get('threshold', 0.1)
    
    def detect(self, image: np.ndarray,
               threshold: Optional[float] = None) -> Tuple[bool, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        high_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, 
                                              center_w-w//4:center_w+w//4]
        
        high_freq_energy = np.sum(high_freq_region)
        total_energy = np.sum(magnitude_spectrum)
        
        high_freq_ratio = high_freq_energy / total_energy
        
        if threshold is None:
            threshold = self.threshold
        
        is_blurry = high_freq_ratio < threshold
        
        self.logger.info(f"FFT high freq ratio: {high_freq_ratio:.4f}, threshold: {threshold}")
        
        return is_blurry, float(high_freq_ratio)
    
    def get_magnitude_spectrum(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        return magnitude_spectrum
    
    def get_phase_spectrum(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        phase_spectrum = np.angle(fshift)
        
        return phase_spectrum
    
    def analyze_frequency_distribution(self, image: np.ndarray) -> dict:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        regions = {
            'low': magnitude[center_h-h//8:center_h+h//8, 
                            center_w-w//8:center_w+w//8],
            'medium': magnitude[center_h-h//4:center_h+h//4, 
                               center_w-w//4:center_w+w//4],
            'high': magnitude
        }
        
        energies = {}
        for name, region in regions.items():
            energies[name] = float(np.sum(region))
        
        total_energy = energies['high']
        ratios = {
            'low': energies['low'] / total_energy,
            'medium': energies['medium'] / total_energy,
            'high': 1.0
        }
        
        return {'energies': energies, 'ratios': ratios}
    
    def detect_motion_blur_direction(self, image: np.ndarray) -> Tuple[float, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        angles = np.linspace(0, np.pi, 180)
        strengths = []
        
        for angle in angles:
            theta = np.deg2rad(angle)
            length = min(h, w) // 4
            
            y_coords = center_h + length * np.sin(theta) * np.linspace(-1, 1, 100)
            x_coords = center_w + length * np.cos(theta) * np.linspace(-1, 1, 100)
            
            y_coords = np.clip(y_coords, 0, h-1).astype(int)
            x_coords = np.clip(x_coords, 0, w-1).astype(int)
            
            strength = np.mean(magnitude[y_coords, x_coords])
            strengths.append(strength)
        
        strengths = np.array(strengths)
        min_strength_idx = np.argmin(strengths)
        blur_direction = angles[min_strength_idx]
        blur_strength = 1.0 - (strengths[min_strength_idx] / strengths.max())
        
        return float(blur_direction), float(blur_strength)
