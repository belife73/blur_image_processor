import cv2
import numpy as np
from typing import Dict, List, Optional
from utils.logger import Logger


class ImageStatistics:
    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger()
    
    def basic_statistics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        stats = {
            'mean': float(gray.mean()),
            'median': float(np.median(gray)),
            'std': float(gray.std()),
            'variance': float(gray.var()),
            'min': float(gray.min()),
            'max': float(gray.max()),
            'range': float(gray.max() - gray.min())
        }
        
        return stats
    
    def histogram_statistics(self, image: np.ndarray) -> Dict[str, any]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel()
        
        total_pixels = hist.sum()
        cumulative = np.cumsum(hist) / total_pixels
        
        stats = {
            'histogram': hist,
            'cumulative': cumulative,
            'peak_intensity': int(np.argmax(hist)),
            'peak_count': int(hist.max()),
            'median_intensity': int(np.searchsorted(cumulative, 0.5))
        }
        
        return stats
    
    def channel_statistics(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        if len(image.shape) != 3:
            raise ValueError("Image must be color (3 channels)")
        
        channels = cv2.split(image)
        channel_names = ['B', 'G', 'R']
        
        stats = {}
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            stats[name] = {
                'mean': float(channel.mean()),
                'std': float(channel.std()),
                'min': float(channel.min()),
                'max': float(channel.max())
            }
        
        return stats
    
    def gradient_statistics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        stats = {
            'gradient_mean': float(gradient_magnitude.mean()),
            'gradient_std': float(gradient_magnitude.std()),
            'gradient_max': float(gradient_magnitude.max()),
            'edge_pixels': int(np.sum(gradient_magnitude > 50)),
            'edge_ratio': float(np.sum(gradient_magnitude > 50) / gradient_magnitude.size),
            'dominant_direction': float(np.median(gradient_direction))
        }
        
        return stats
    
    def frequency_statistics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        low_freq = magnitude[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8]
        high_freq = magnitude
        
        stats = {
            'dc_component': float(magnitude[center_h, center_w]),
            'low_freq_energy': float(np.sum(low_freq)),
            'total_energy': float(np.sum(high_freq)),
            'high_freq_ratio': float(1.0 - np.sum(low_freq) / np.sum(high_freq)),
            'spectral_centroid': self._calculate_spectral_centroid(magnitude)
        }
        
        return stats
    
    def _calculate_spectral_centroid(self, magnitude: np.ndarray) -> float:
        h, w = magnitude.shape
        y_coords, x_coords = np.indices((h, w))
        
        total_energy = np.sum(magnitude)
        if total_energy == 0:
            return 0.0
        
        centroid_x = np.sum(x_coords * magnitude) / total_energy
        centroid_y = np.sum(y_coords * magnitude) / total_energy
        
        distance = np.sqrt((centroid_x - w/2)**2 + (centroid_y - h/2)**2)
        
        return float(distance)
    
    def texture_statistics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        glcm = self._calculate_glcm(gray)
        
        stats = {
            'contrast': self._calculate_contrast(glcm),
            'homogeneity': self._calculate_homogeneity(glcm),
            'energy': self._calculate_energy(glcm),
            'correlation': self._calculate_correlation(glcm)
        }
        
        return stats
    
    def _calculate_glcm(self, image: np.ndarray, distance: int = 1) -> np.ndarray:
        h, w = image.shape
        glcm = np.zeros((256, 256))
        
        for i in range(h - distance):
            for j in range(w - distance):
                glcm[image[i, j], image[i + distance, j + distance]] += 1
        
        glcm = glcm / glcm.sum()
        
        return glcm
    
    def _calculate_contrast(self, glcm: np.ndarray) -> float:
        i, j = np.indices(glcm.shape)
        contrast = np.sum(glcm * (i - j) ** 2)
        return float(contrast)
    
    def _calculate_homogeneity(self, glcm: np.ndarray) -> float:
        i, j = np.indices(glcm.shape)
        homogeneity = np.sum(glcm / (1 + np.abs(i - j)))
        return float(homogeneity)
    
    def _calculate_energy(self, glcm: np.ndarray) -> float:
        energy = np.sum(glcm ** 2)
        return float(energy)
    
    def _calculate_correlation(self, glcm: np.ndarray) -> float:
        i, j = np.indices(glcm.shape)
        
        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)
        
        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        
        if std_i == 0 or std_j == 0:
            return 0.0
        
        correlation = np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
        return float(correlation)
    
    def noise_statistics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        m, n = int(h * 0.1), int(w * 0.1)
        
        noise_region = gray[m:h-m, n:w-n]
        
        noise_mean = noise_region.mean()
        noise_std = noise_region.std()
        
        stats = {
            'noise_mean': float(noise_mean),
            'noise_std': float(noise_std),
            'snr': float(gray.mean() / (noise_std + 1e-6)),
            'noise_level': float(noise_std / 255.0)
        }
        
        return stats
    
    def comprehensive_analysis(self, image: np.ndarray) -> Dict[str, any]:
        analysis = {
            'basic': self.basic_statistics(image),
            'histogram': self.histogram_statistics(image),
            'gradient': self.gradient_statistics(image),
            'frequency': self.frequency_statistics(image),
            'noise': self.noise_statistics(image)
        }
        
        if len(image.shape) == 3:
            analysis['channels'] = self.channel_statistics(image)
        
        return analysis
    
    def compare_statistics(self, image1: np.ndarray,
                            image2: np.ndarray) -> Dict[str, float]:
        stats1 = self.basic_statistics(image1)
        stats2 = self.basic_statistics(image2)
        
        comparison = {}
        for key in stats1.keys():
            comparison[f'{key}_diff'] = abs(stats1[key] - stats2[key])
            comparison[f'{key}_ratio'] = stats2[key] / (stats1[key] + 1e-6)
        
        return comparison
